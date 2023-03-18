import argparse
import os
import random
import torch
import numpy as np
import pandas as pd
import warnings
import gc
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

from torchvision import transforms
from torch.utils.data import DataLoader
from focal_loss import FocalLoss
import torch.optim as optim
import torch.nn as nn
import wandb

import datasets
import models
import utils
import training
import evaluation
import preprocessing


warnings.filterwarnings('ignore')

ROOT_DIR = '.'
DATA_DIR = 'data'
SUBMISSION_DIR = 'submit'
MODEL_DIR = 'model'
PRETRAINED_DIR = 'pretrained'

CFG = {
    'VIDEO_LENGTH': 50,      # 10프레임 * 5초
    'IMG_HEIGHT': 255,
    'IMG_WIDTH': 255,
    'EPOCHS': 20,            # Epochs
    'LEARNING_RATE': 3e-4,   # Learning Rate
    'FRAME_SKIP': 1,
    'BATCH_SIZE': 8,         # batch_size
    'SEED': 42,              # SEED 값
    'NAME': 'Classifier-V2',
    'CUDA': 'cuda:1', 
    'TARGET': 'weather',
    'GAMMA': 3,
    'WANDB': 0,
    'MODEL': 'cnn',          # cnn or rnn
    'SAVE_EVERY_EPOCH': 0,
    'EARLY_STOP': 5,         # patience
    'NUM_WORKERS': 8, 
    'N_SPLIT': 3,
    'KFOLD': 1,              # 1: True, 0: False
    'RANDOM_SEED': 0,        # 1: True, 0: False
    'HIDDEN_SIZE': 128,
    'DR_RATE': 0.25,
}

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def create_name(config, fold=0):
    name = f"{config['name']}-fs-{config['frame_skip']}-bs-{config['batch_size']}-hidden-{config['hidden_size']}-" \
           f"lr-{config['lr']:.5f}-dr-{config['dr_rate']:.2f}-gamma-{config['gamma']}-fold-{fold}"
    return name


def main(args):
    CFG['EPOCHS'] = args.epoch
    CFG['BATCH_SIZE'] = args.bs
    CFG['LEARNING_RATE'] = args.lr
    CFG['IMG_HEIGHT'] = args.height
    CFG['IMG_WIDTH'] = args.width
    CFG['NAME'] = args.name
    CFG['SEED'] = args.seed
    CFG['FRAME_SKIP'] = args.fs
    CFG['CUDA'] = f'cuda:{int(args.cuda)}'
    CFG['TARGET'] = args.target
    CFG['GAMMA'] = args.gamma
    CFG['WANDB'] = args.wandb
    CFG['MODEL'] = args.model
    CFG['EARLY_STOP'] = args.es
    CFG['KFOLD'] = args.kfold
    CFG['RANDOM_SEED'] = args.rs
    CFG['HIDDEN_SIZE'] = args.hidden
    CFG['DR_RATE'] = args.dr
    
    if CFG['RANDOM_SEED']:
        print('Generate Random Seed')
        CFG['SEED'] = np.random.randint(1000)
        
    print(CFG)

    CFG['MODEL_NAME'] = f"{CFG['TARGET'].upper()}-{CFG['NAME']}-model-{CFG['MODEL']}-{CFG['IMG_HEIGHT']}x{CFG['IMG_WIDTH']}-fs-{CFG['FRAME_SKIP']}-batch-{CFG['BATCH_SIZE']}-seed-{CFG['SEED']}-kfold-{CFG['KFOLD']}"
    
    print("model name: ", CFG['MODEL_NAME'])
    print('name: ', CFG['NAME'])
    
    # Seed 고정
    seed_everything(CFG['SEED']) 
    
    # pre-processing
    train = preprocessing.load('.', 'data', 'train.csv')
    train = preprocessing.create_features(train)
    test = preprocessing.load('.', 'data', 'test.csv')
    
    # 오류 데이터 처리 - 데이콘 토크 참조
    exempt_ids = [
        'TRAIN_0486',
        'TRAIN_0124',
        'TRAIN_0387',
        'TRAIN_2292',
        'TRAIN_0008',
        'TRAIN_0330',
        'TRAIN_1113',
        'TRAIN_0144' # 중복 데이터데이터
    ]
    train = train.loc[~train['sample_id'].isin(exempt_ids)].reset_index(drop=True)

    kfold = StratifiedKFold(n_splits=CFG['N_SPLIT'], 
                        shuffle=True,
                        random_state=CFG['SEED'])
        
    if CFG['TARGET'] == 'crash':
        train_crash, valid_crash = train_test_split(train, 
                                                    test_size=0.2,
                                                    stratify=train[CFG['TARGET']],
                                                    random_state=CFG['SEED'])
        
        train_crash = train_crash.reset_index(drop=True)
        valid_crash = valid_crash.reset_index(drop=True)
        
        for fold, (_, test_idx) in enumerate(kfold.split(train_crash, train_crash[CFG['TARGET']])):
            train_crash.loc[test_idx, 'fold'] = fold
        kfold_data = train_crash.copy()
            
        
    else:
        # extract only crash data
        crash_data = train.loc[train['label'] != 0].reset_index(drop=True)
        
        train_crash, valid_crash = train_test_split(crash_data, 
                                                    test_size=0.2,
                                                    stratify=crash_data[CFG['TARGET']],
                                                    random_state=CFG['SEED'])
        
        train_crash = train_crash.reset_index(drop=True)
        valid_crash = valid_crash.reset_index(drop=True)
        
        for fold, (_, test_idx) in enumerate(kfold.split(train_crash, train_crash[CFG['TARGET']])):
            train_crash.loc[test_idx, 'fold'] = fold
            
        kfold_data = train_crash.copy()             
    
    config = {
        'lr' : CFG['LEARNING_RATE'],
        'batch_size' : CFG['BATCH_SIZE'],
        'hidden_size': CFG['HIDDEN_SIZE'],
        'dr_rate': CFG['DR_RATE'],
        'epochs': CFG['EPOCHS'],
        'early_stop': CFG['EARLY_STOP'],
        'frame_skip': CFG['FRAME_SKIP'],
        'gamma': CFG['GAMMA'],   
        'name': CFG['NAME'],
        'resize': 240 if CFG['IMG_HEIGHT'] == 224 else 120,
        'crop': 224 if CFG['IMG_HEIGHT'] == 224 else 112,
    }
    
    CFG['IMG_WIDTH'] = config['crop']
    CFG['IMG_HEIGHT'] = config['crop']
    
    transform = dict()
    transform['aug'] = transforms.Compose([
        transforms.ToPILImage(),
        # transforms.Resize(size=(120, 120)),
        # transforms.CenterCrop(size=(112, 112)),
        transforms.Resize(size=(config['resize'], config['resize'])),
        transforms.CenterCrop(size=(config['crop'], config['crop'])),
        # transforms.RandomCrop(size=(224, 224)),
        # transforms.FiveCrop(size=(360, 640)),
        # transforms.Lambda(lambda crops: torch.stack([transforms.PILToTensor()(crop) for crop in crops])),
        # transforms.Lambda(lambda crops: torch.stack([transforms.Resize(size=(CFG['IMG_HEIGHT'], CFG['IMG_WIDTH']))(crop) for crop in crops])),
        # transforms.Lambda(lambda crops: torch.stack([transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(crop) for crop in crops])),
        # transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
        # transforms.RandomEqualize(p=1.0),
        # transforms.CenterCrop(size=(360, 640)),
        # transforms.RandomAutocontrast(p=1.0),
        # transforms.RandomSolarize(200, p=1.0),
        # transforms.RandomInvert(p=1.0),
        # transforms.RandomEqualize(p=1.0),
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        # transforms.Normalize((0.43216, 0.394666, 0.37645), (0.22803, 0.22145, 0.216989)),
    ])
  
    transform['normal'] = transforms.Compose([
        transforms.ToPILImage(),
        # transforms.Resize(size=(120, 120)),
        # transforms.CenterCrop(size=(112, 112)),
        transforms.Resize(size=(config['resize'], config['resize'])),
        transforms.CenterCrop(size=(config['crop'], config['crop'])),
        # transforms.CenterCrop(size=(CFG['IMG_HEIGHT'], CFG['IMG_WIDTH'])),
        # transforms.CenterCrop(size=(480, 800)),
        # transforms.Resize(size=(CFG['IMG_HEIGHT'], CFG['IMG_WIDTH'])),
        # transforms.Resize(size=(CFG['IMG_HEIGHT'], CFG['IMG_WIDTH'])),
        
        # transforms.RandomInvert(p=1.0),
        # transforms.RandomEqualize(p=1.0),
        # transforms.RandomEqualize(p=1.0),
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        # transforms.Normalize(0.45, 0.225)
        # transforms.Normalize((0.43216, 0.394666, 0.37645), (0.22803, 0.22145, 0.216989)),
    ])
    
    # train dataset 생성
    shape = 'c d h w'
    if CFG['MODEL'] == 'attention':
        print('Attention Model')
        shape = 'd c h w'
        
    train_loaders, valid_loaders = [], []
    for fold in range(CFG['N_SPLIT']):
        trn = kfold_data.loc[kfold_data['fold'] != fold]
        vld = kfold_data.loc[kfold_data['fold'] == fold]
        
        trn_ds = datasets.EWTDataset(trn, transform=transform, frame_skip=CFG['FRAME_SKIP'], mode='train', shape=shape)
        trn_loader = DataLoader(trn_ds, 
                                batch_size=CFG['BATCH_SIZE'], 
                                shuffle=True, 
                                num_workers=CFG['NUM_WORKERS'])
    
        train_loaders.append(trn_loader)
        
        vld_ds = datasets.EWTDataset(vld, transform=transform, frame_skip=CFG['FRAME_SKIP'], mode='validation', shape=shape)
        vld_loader = DataLoader(vld_ds, 
                                batch_size=CFG['BATCH_SIZE'], 
                                shuffle=False, 
                                num_workers=CFG['NUM_WORKERS'])
    
        valid_loaders.append(vld_loader)
        
    # test 데이터셋 로드
    test_dataset = datasets.EWTDataset(test, transform=transform, frame_skip=CFG['FRAME_SKIP'], mode='test', shape=shape)
    
    # test dataloader 생성
    test_loader = DataLoader(test_dataset, 
                            batch_size=CFG['BATCH_SIZE'], 
                            shuffle=False, 
                            num_workers=CFG['NUM_WORKERS'])
    
    video_path, x, c, e, w, t, l = next(iter(train_loaders[0]))
    print('x.shape: ', x.shape)
    
    device = torch.device(CFG['CUDA']) if torch.cuda.is_available() else torch.device('cpu')
    print('device: ', device)
    
    weight_data = kfold_data.copy()
    
    # calculate weights
    crash_weight = utils.get_class_weights(weight_data['crash'])
    ego_weight = utils.get_class_weights(weight_data['ego'])
    weather_weight = utils.get_class_weights(weight_data['weather'])
    timing_weight = utils.get_class_weights(weight_data['timing'])

    if CFG['TARGET'].lower() == 'crash':
        # 베이스라인 모델 생성
        num_classes = 2
        weights = crash_weight
    elif CFG['TARGET'].lower() == 'ego':
        # 베이스라인 모델 생성
        num_classes = 2
        weights = ego_weight

    elif CFG['TARGET'].lower() == 'weather':
        num_classes = 3
        weights = weather_weight
        
    elif CFG['TARGET'].lower() == 'timing':
        num_classes = 2
        weights = timing_weight
        
    weights = utils.compute_pos_weights(torch.Tensor(weights))
    
    print('CFG', CFG)
    
    NUM_EPOCHS = CFG['EPOCHS']

    fold_losses, fold_scores, fold_preds = [], [], []

    
    if CFG['WANDB']:

        # wandb setup
        wandb.init(project='Dacon Car Crash', 
                   group=f"{CFG['TARGET'].upper()}",
                   name=create_name(config),
                   notes=f"{CFG['MODEL']}-SEED-{CFG['SEED']}-KFOLD-{CFG['KFOLD']}-EarlyStop-{CFG['EARLY_STOP']}-NUM_EPOCH-{CFG['EPOCHS']}",
                   tags=f"{CFG['MODEL']}",
                   entity='teddynote')
        wandb.define_metric('train_loss', summary='min')
        wandb.define_metric('val_loss', summary='min')
        wandb.define_metric('fold_loss', summary='min')
        wandb.define_metric('train_f1', summary='max')
        wandb.define_metric('val_f1', summary='max')
        wandb.define_metric('fold_f1', summary='max')
        wandb.config = args
    
    # kfold training
    for idx, (trn_loader, vld_loader) in enumerate(zip(train_loaders, valid_loaders)):
            # model 설정    
        if CFG['NAME'] == 'rnn':
            print('Using RNN based Model')
            model = models.Resnt18Rnn(num_classes=num_classes, rnn_hidden_size=config['hidden_size'], drop_rate=config['dr_rate']).to(device)
        else:
            print('Using Conv3D based Model')
            model = models.EWTClassifier(num_classes=num_classes, hidden_size=config['hidden_size'], dr_rate=config['dr_rate'], model_name=config['name']).to(device)
        
        # Early Stopping
        early_stopping = utils.EarlyStopping(patience=CFG['EARLY_STOP'], verbose=True, mode='min')
        
        # optimizer
        optimizer = optim.Adam(model.parameters(), lr=config['lr'])
        # optimizer = optim.AdamW(model.parameters(), lr=CFG["LEARNING_RATE"], weight_decay=0.001)
        
        # loss
        loss = FocalLoss(alpha=weights, gamma=config['gamma'], size_average=True, device=device)
        
        # Learning Rate Scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                mode='max', 
                                                                factor=0.5, 
                                                                patience=2,
                                                                threshold_mode='abs',
                                                                min_lr=1e-8, 
                                                                verbose=True)
        
        min_loss = np.inf
        max_f1 = 0
        
        if CFG['WANDB']:
            wandb.watch(model)
        
        print(f'[INFO] KFold: {idx+1}/{len(train_loaders)}')
        wandb.config.name = create_name(config, fold=idx)
        
        # Epoch 별 훈련 및 검증을 수행합니다.
        for epoch in range(NUM_EPOCHS):
            # Model Training
            # 훈련 손실과 정확도를 반환 받습니다.
            t_loss, t_score, train_incorr = training.ewt_classifier_train(model, trn_loader, loss, optimizer, target=CFG['TARGET'], device=device)
            
            # 검증 손실과 검증 정확도를 반환 받습니다.
            v_loss, v_score, val_incorr = evaluation.ewt_classifier_evaluate(model, vld_loader, loss, target=CFG['TARGET'], device=device)
            
            print(f"Epoch: {epoch+1}, Fold: {idx+1}, loss: {t_loss:.4f}, f1: {t_score:.4f}, val_loss: {v_loss:.4f}, val_f1: {v_score:.4f}")
            
                # val_loss 가 개선되었다면 min_loss를 갱신하고 model의 가중치(weights)를 저장합니다.
            if v_loss < min_loss:
                print(f'[INFO] val_loss has been improved from {min_loss:.5f} to {v_loss:.5f}. Saving Model!')
                min_loss = v_loss
                max_f1 = v_score
                torch.save(model.state_dict(), os.path.join(MODEL_DIR, f"{CFG['MODEL_NAME']}-fold-{idx}-{min_loss:.5f}-f1-{max_f1:.5f}.pth"))
            
            # log per epoch
            if CFG['WANDB']:
                wandb.log({'fold': idx, 'loss': t_loss, 'val_loss': v_loss, 'train_f1': t_score, 'val_f1': v_score}, step=epoch)
                
            # early stopping check (mode='min') val_loss 기준으로 측정
            early_stopping(score=v_loss)
            if early_stopping.early_stop:
                break
            
            # scheduler
            scheduler.step(v_loss)
                
        # inference
        # 모델에 저장한 가중치를 로드합니다.
        model.load_state_dict(torch.load(os.path.join(MODEL_DIR, f"{CFG['MODEL_NAME']}-fold-{idx}-{min_loss:.5f}-f1-{max_f1:.5f}.pth")))
        # evaluation
        fold_pred = evaluation.inference_proba(model, test_loader, device)
        
        # fold 예측치 추가
        fold_preds.append(fold_pred)
        
        # Fold의 최종성능 결과를 출력합니다.
        print(f'============= Fold {idx:02d} =============')
        print(f'Fold {idx:02d}, val_loss: {min_loss:.5f}, val_f1: {max_f1:.5f}')
        
        # log per fold
        if CFG['WANDB']:
            wandb.log({'fold': idx, 'val_loss': min_loss, 'val_f1': max_f1})
        print(f'==========================================')
           
        fold_losses.append(min_loss)
        fold_scores.append(max_f1)
      
    fold_mean_loss = sum(fold_losses) / len(fold_losses)
    fold_mean_score = sum(fold_scores) / len(fold_scores)              
    
    # save to pickle
    PICKLE_FILENAME = f"[XVAL]{CFG['MODEL_NAME']}-{fold_mean_loss:.5f}-f1-{fold_mean_score:.5f}.pkl"
    utils.save_pickle(fold_preds, pickle_path='pickle', filename=PICKLE_FILENAME)
    
    wandb.log({'fold_loss': fold_mean_loss, 'fold_f1': fold_mean_score})
    
    if CFG['WANDB']:
        wandb.alert('Training Task Finished', f"{PICKLE_FILENAME} / VAL_LOSS: {fold_mean_loss:.5f}, F1 SCORE: {fold_mean_score:.5f}")
        
    gc.collect()
    torch.cuda.empty_cache()
    
    
if __name__ == '__main__':
    # 인자값을 받을 수 있는 인스턴스 생성
    parser = argparse.ArgumentParser(description='Argparse Tutorial')
    # 입력받을 인자값 설정 (default 값 설정가능)
    parser.add_argument('--epoch',          type=int,   default=CFG['EPOCHS'])
    parser.add_argument('--bs',             type=int,   default=CFG['BATCH_SIZE'])
    parser.add_argument('--lr',             type=float, default=CFG['LEARNING_RATE'])
    parser.add_argument('--height',         type=int,   default=CFG['IMG_HEIGHT'])
    parser.add_argument('--width',          type=int,   default=CFG['IMG_WIDTH'])
    parser.add_argument('--name',                       default=CFG['NAME'])
    parser.add_argument('--seed',           type=int,   default=CFG['SEED'])
    parser.add_argument('--fs',             type=int,   default=CFG['FRAME_SKIP'])
    parser.add_argument('--cuda',           type=int,   default=0)
    parser.add_argument('--target',                     default=CFG['TARGET'])
    parser.add_argument('--gamma',          type=int,   default=CFG['GAMMA'])
    parser.add_argument('--wandb',          type=int,   default=CFG['WANDB'])
    parser.add_argument('--model',                      default=CFG['MODEL'])
    # parser.add_argument('--save',           type=int,   default=CFG['SAVE_EVERY_EPOCH'])
    parser.add_argument('--es',             type=int,   default=CFG['EARLY_STOP'])
    parser.add_argument('--kfold',          type=int,   default=CFG['KFOLD'])
    parser.add_argument('--rs',             type=int,   default=CFG['RANDOM_SEED'])
    parser.add_argument('--hidden',         type=int,   default=CFG['HIDDEN_SIZE'])
    parser.add_argument('--dr',             type=float, default=CFG['DR_RATE'])
    
    # args 에 위의 내용 저장
    args    = parser.parse_args()
    main(args=args)
    
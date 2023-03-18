import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score
import torch


def ewt_classifier_evaluate(model, data_loader, loss_fn, target, device):
    # model.eval()은 모델을 평가모드로 설정을 바꾸어 줍니다. 
    # dropout과 같은 layer의 역할 변경을 위하여 evaluation 진행시 꼭 필요한 절차 입니다.
    model.eval()
    
    trues, preds = [], []
    
    incorrect = {
        'video_path': [],
        'label': [],
        'pred': [],
    }
    
    # Gradient가 업데이트 되는 것을 방지 하기 위하여 반드시 필요합니다.
    with torch.no_grad():
        # loss와 accuracy 계산을 위한 임시 변수 입니다. 0으로 초기화합니다.
        corr = 0
        running_loss = 0
        running_size = 0
        
        prograss_bar = tqdm(data_loader)
        
        # 배치별 evaluation을 진행합니다.
        for batch_idx, (video_path, video, crash, ego, weather, timing, lbl) in enumerate(prograss_bar, start=1):
            # image, label 데이터를 device에 올립니다.
            if target == 'crash':
                video, lbl = video.to(device), crash.to(device)
            elif target == 'ego':
                video, lbl = video.to(device), ego.to(device)
            elif target == 'weather' or target == 'label':
                video, lbl = video.to(device), weather.to(device)
            elif target == 'timing':
                video, lbl = video.to(device), timing.to(device)
            
            # 모델에 Forward Propagation을 하여 결과를 도출합니다.
            output = model(video)

            # 손실함수에 output, label 값을 대입하여 손실을 계산합니다.
            loss = loss_fn(output, lbl)
            
            pred = output.argmax(dim=1)

            # loss 값은 1개 배치의 평균 손실(loss) 입니다. video.size(0)은 배치사이즈(batch size) 입니다.
            # loss 와 video.size(0)를 곱하면 1개 배치의 전체 loss가 계산됩니다.
            # 이를 누적한 뒤 Epoch 종료시 전체 데이터셋의 개수로 나누어 평균 loss를 산출합니다.
            running_loss += loss.item() * video.size(0)
            running_size += video.size(0)
            
            batch_lbls = lbl.detach().cpu().numpy()
            batch_preds = pred.detach().cpu().numpy()
            
            trues += batch_lbls.tolist()
            preds += batch_preds.tolist()
            
            incorrect_idx = lbl.ne(pred).detach().cpu().numpy()
            incorrect_lbl = batch_lbls[incorrect_idx]
            video_path = np.array(video_path)
            incorrect_video = video_path[incorrect_idx]
            
            incorrect['label'].append(incorrect_lbl.tolist())
            incorrect['video_path'].append(incorrect_video.tolist())
            
            # 배치별 f1 score를 계산합니다.
            f1 = f1_score(batch_lbls, batch_preds, average='macro')
            # Accuracy 계산
            acc = accuracy_score(batch_lbls, batch_preds)
            
            prograss_bar.set_description(f'[Evaluation] loss: {running_loss / running_size:.4f}, f1 score: {f1:.4f}, accuracy: {acc:.4f}')
        
        f1 = f1_score(trues, preds, average='macro')
        
        # 결과를 반환합니다.
        # val_loss, f1 score
        return running_loss / len(data_loader.dataset), f1, incorrect
    
    
    # 추론을 위한 함수
def inference(model, test_loader, device):
    model.to(device)
    model.eval()
    preds = []
    with torch.no_grad():
        for video in tqdm(test_loader):
            video = video.to(device)
            e, w, t = model(video)
            
            ego_pred = e.argmax(dim=1)
            weather_pred = w.argmax(dim=1)
            timing_pred = t.argmax(dim=1)

            pred = torch.add(6*ego_pred, 2*weather_pred)
            pred = torch.add(pred, timing_pred)
            pred = torch.add(pred, 1)
            preds += pred.detach().cpu().numpy().tolist()
    return preds

# 확률 추론을 위한 함수
def inference_proba(model, test_loader, device):
    model.to(device)
    model.eval()
    outputs = []
    with torch.no_grad():
        for path, video in tqdm(test_loader):
            video = video.to(device)
            output = model(video)

            outputs += output.detach().cpu().numpy().tolist()
    return outputs
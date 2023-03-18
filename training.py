import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score


def ewt_classifier_train(model, data_loader, loss_fn, optimizer, target, device):
    # 모델을 훈련모드로 설정합니다. training mode 일 때 Gradient 가 업데이트 됩니다. 반드시 train()으로 모드 변경을 해야 합니다.
    model.train()
    
    # loss와 accuracy 계산을 위한 임시 변수 입니다. 0으로 초기화합니다.
    running_size = 0
    running_loss = 0
    
    # trues, preds
    trues, preds = [], []
    
    # 예쁘게 Progress Bar를 출력하면서 훈련 상태를 모니터링 하기 위하여 tqdm으로 래핑합니다.
    prograss_bar = tqdm(data_loader)
    
    incorrect = {
        'video_path': [],
        'label': []
    }
    
    # mini-batch 학습을 시작합니다.
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
        
        # 누적 Gradient를 초기화 합니다.
        optimizer.zero_grad()
        
        # Forward Propagation을 진행하여 결과를 얻습니다.
        output = model(video)
        
        # 손실함수에 output, label 값을 대입하여 손실을 계산합니다.
        loss = loss_fn(output, lbl)
        
        # 오차역전파(Back Propagation)을 진행하여 미분 값을 계산합니다.
        loss.backward()
        
        # 계산된 Gradient를 업데이트 합니다.
        optimizer.step()
        
        # output의 max(dim=1)은 max probability와 max index를 반환합니다.
        # max probability는 무시하고, max index는 pred에 저장하여 label 값과 대조하여 정확도를 도출합니다.
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
        
        # print('===')
        # print(incorrect_lbl.tolist())
        # print(incorrect_video.tolist())
        # print(batch_preds[incorrect_idx].tolist())
        # print(batch_lbls.tolist())
        # print(batch_preds.tolist())
        # print(video_path.tolist())
        # print('===')
        
        # 배치별 f1 score를 계산합니다.
        f1 = f1_score(batch_lbls, batch_preds, average='macro')
        # Accuracy 계산
        acc = accuracy_score(batch_lbls, batch_preds)
        
        prograss_bar.set_description(f'[Training] loss: {running_loss / running_size:.4f}, f1 score: {f1:.4f}, accuracy: {acc:.4f}')
    
    # 최종 f1 score를 계산합니다.
    f1 = f1_score(trues, preds, average='macro')
    
    # 평균 손실(loss)를 반환합니다.
    # train_loss
    return running_loss / len(data_loader.dataset), f1, incorrect
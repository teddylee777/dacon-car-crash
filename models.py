import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import models
import torch


class BaseClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super(BaseClassifier, self).__init__()
        self.feature_extract = nn.Sequential(
            nn.Conv3d(3, 8, (1, 3, 3)),
            nn.ReLU(),
            nn.BatchNorm3d(8),
            nn.MaxPool3d(2),
            nn.Conv3d(8, 32, (1, 2, 2)),
            nn.ReLU(),
            nn.BatchNorm3d(32),
            nn.MaxPool3d(2),
            nn.Conv3d(32, 64, (1, 2, 2)),
            nn.ReLU(),
            nn.BatchNorm3d(64),
            nn.MaxPool3d(2),
            nn.Conv3d(64, 128, (1, 2, 2)),
            nn.ReLU(),
            nn.BatchNorm3d(128),
            nn.MaxPool3d((3, 7, 7)),
        )
        self.classifier = nn.Linear(1024, num_classes)
        
    def forward(self, x):
        batch_size = x.size(0)
        x = self.feature_extract(x)
        x = x.view(batch_size, -1)
        x = self.classifier(x)
        return x    
    
class EWTClassifier(nn.Module):
    def __init__(self, num_classes, dr_rate=0.25, hidden_size=128, model_name='x3d_m'):
        super(EWTClassifier, self).__init__()
        if model_name == 'x3d_m':
            self.net = torch.hub.load('facebookresearch/pytorchvideo', 'x3d_m', pretrained=True)
        elif model_name == 'slow_fast':
            self.net = torch.hub.load('facebookresearch/pytorchvideo', 'slowfast_r50', pretrained=True)
        elif model_name == 'r2plus1d_18':
            self.net = models.video.r2plus1d_18(pretrained=True)
        elif model_name == 'r3d_18':
            self.net = models.video.r3d_18(pretrained=True)
        elif model_name == 'mc3_18':
            self.net = models.video.mc3_18(pretrained=True)
                
        self.fc = nn.Sequential(
            nn.Dropout(dr_rate),
            nn.Linear(400, hidden_size), 
            nn.ReLU(),
            nn.Dropout(dr_rate),
        )
        
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.net(x)
        x = x.view(batch_size, -1)
        x = self.fc(x)
        x = self.classifier(x)
        return F.softmax(x, dim=1)
    
    
class Resnt18Rnn(nn.Module):
    def __init__(self, num_classes, drop_rate=0.25, rnn_hidden_size=128):
        super(Resnt18Rnn, self).__init__()
        num_classes = num_classes
        dr_rate = drop_rate
        rnn_num_layers = 2
        num_features = 128
        
        baseModel = models.resnet50(pretrained=True)            
        num_features = baseModel.fc.in_features
        
        baseModel.fc = Identity()
        self.baseModel = baseModel
        self.dropout= nn.Dropout(dr_rate)
        self.rnn = nn.LSTM(num_features, rnn_hidden_size, rnn_num_layers, bidirectional=True)
        self.fc1 = nn.Linear(rnn_hidden_size*2, num_classes)
        
    def forward(self, x):
        b_z, c, ts, h, w = x.shape
        ii = 0
        y = self.baseModel((x[:, :, ii]))
        output, (hn, cn) = self.rnn(y.unsqueeze(1))
        for ii in range(1, ts):
            y = self.baseModel((x[:, :, ii]))
            out, (hn, cn) = self.rnn(y.unsqueeze(1), (hn, cn))
        out = self.dropout(out[:,-1])
        out = self.fc1(out) 
        return out 
    
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x   
    


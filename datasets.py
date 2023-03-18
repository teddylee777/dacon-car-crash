import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
import numpy as np


class EWTDatasetCV(Dataset):
    def __init__(self, dataframe, imgH, imgW, frame_count=5*10, frame_skip=1, mode='train'):
        super(EWTDatasetCV, self).__init__()
        self.data = dataframe
        self.frame_count = frame_count
        self.mode = mode
        self.read_every = frame_skip
        self.img_size = (imgH, imgW)
        
    def __getitem__(self, idx):
        video_path = self.data.iloc[idx]['video_path']
        frames = self.load_video(video_path)
        if self.mode == 'train' or self.mode == 'validation':
            label = self.data.iloc[idx]['label']

            # ego-involve
            # 0: yes, 1: no
            if label > 0 and label <= 6:
                ego = 0
            else:
                ego = 1

            # weather
            # 0: normal, 1: snowy, 2: rainy
            if label in [1, 2, 7, 8]:
                weather = 0
            elif label in [3, 4, 9, 10]:
                weather = 1
            elif label in [5, 6, 11, 12]:
                weather = 2

            # timing
            # 0: day, 1: night
            if label in [1, 3, 5, 7, 9, 11]:
                timing = 0
            elif label in [2, 4, 6, 8, 10, 12]:
                timing = 1
            
            return video_path, frames, ego, weather, timing, label
        else:
            return video_path, frames
        
    def __len__(self):
        return len(self.data)
    
    # 영상 로드
    def load_video(self, video_path):
        frames = []
        cap = cv2.VideoCapture(video_path)
        for idx, _ in enumerate(range(self.frame_count)):
            _, img = cap.read()
            if idx % self.read_every == 0:
                img = cv2.resize(img, (self.img_size[1], self.img_size[0]))
                img = img / 255.
                frames.append(img)

            # depth, height, width, channel => channel, depth, height, width 로 순서 변경
            # (50, 224, 224, 3)             => (3, 50, 224, 224)
            # Conv3D 에 입력으로 넣기 위함
        return torch.FloatTensor(np.array(frames)).permute(3, 0, 1, 2)


class EWTDataset(Dataset):
    def __init__(self, dataframe, transform, frame_count=5*10, frame_skip=1, mode='train', shape='c d h w'):
        super(EWTDataset, self).__init__()
        self.data = dataframe
        self.frame_count = frame_count
        self.mode = mode
        self.read_every = frame_skip
        self.transform = transform
        self.shape = shape
        
    def __getitem__(self, idx):
        video_path = self.data.iloc[idx]['video_path']
        frames = self.load_video(video_path)
        if self.mode == 'train' or self.mode == 'validation':
            label = self.data.iloc[idx]['label']

            # crash
            # 0: no, 1: yes
            if label == 0:
                crash = 0
            else:
                crash = 1
                
            # ego-involve
            # 0: yes, 1: no
            if label >= 1 and label <= 6:
                ego = 0
            elif label >= 7 and label <= 12:
                ego = 1
            else:
                ego = -1
            

            # weather
            # 0: normal, 1: snowy, 2: rainy
            if label in [1, 2, 7, 8]:
                weather = 0
            elif label in [3, 4, 9, 10]:
                weather = 1
            elif label in [5, 6, 11, 12]:
                weather = 2
            else:
                weather = -1

            # timing
            # 0: day, 1: night
            if label in [1, 3, 5, 7, 9, 11]:
                timing = 0
            elif label in [2, 4, 6, 8, 10, 12]:
                timing = 1
            else: 
                timing = -1
            
            return video_path, frames, crash, ego, weather, timing, label
        else:
            return video_path, frames
        
    def __len__(self):
        return len(self.data)
    
    # 영상 로드
    def load_video(self, video_path):
        frames = []
        cap = cv2.VideoCapture(video_path)
        for idx, _ in enumerate(range(self.frame_count)):
            _, img = cap.read()
            if idx % self.read_every == 0:
                img = torch.from_numpy(img).permute(2, 0, 1)
                if self.mode == 'train':
                    img = self.transform['aug'](img)
                else:
                    img = self.transform['normal'](img)
                frames.append(img)

            # depth, height, width, channel => channel, depth, height, width 로 순서 변경
            # (50, 224, 224, 3)             => (3, 50, 224, 224)
            # Conv3D 에 입력으로 넣기 위함
        if self.shape == 'c d h w':
            return torch.stack(frames).permute(1, 0, 2, 3)
        elif self.shape == 'd c h w':
            return torch.stack(frames).permute(0, 1, 2, 3)
        
class EWTBinaryDataset(Dataset):
    def __init__(self, dataframe, transform, frame_count=5*10, frame_skip=1, mode='train', shape='c d h w'):
        super(EWTBinaryDataset, self).__init__()
        self.data = dataframe
        self.frame_count = frame_count
        self.mode = mode
        self.read_every = frame_skip
        self.transform = transform
        self.shape = shape
        
    def __getitem__(self, idx):
        video_path = self.data.iloc[idx]['video_path']
        frames = self.load_video(video_path)
        if self.mode == 'train' or self.mode == 'validation':
            label = self.data.iloc[idx]['label']

            # crash
            # 0: no, 1: yes
            if label == 0:
                crash = 0
            else:
                crash = 1
                
            # ego-involve
            # 0: yes, 1: no
            if label >= 1 and label <= 6:
                ego = 0
            elif label >= 7 and label <= 12:
                ego = 1
            else:
                ego = -1
            

            # weather
            # 0: normal, 1: snowy, 2: rainy
            weather = label

            # timing
            # 0: day, 1: night
            if label in [1, 3, 5, 7, 9, 11]:
                timing = 0
            elif label in [2, 4, 6, 8, 10, 12]:
                timing = 1
            else: 
                timing = -1
            
            return video_path, frames, crash, ego, weather, timing, label
        else:
            return video_path, frames
        
    def __len__(self):
        return len(self.data)
    
    # 영상 로드
    def load_video(self, video_path):
        frames = []
        cap = cv2.VideoCapture(video_path)
        for idx, _ in enumerate(range(self.frame_count)):
            _, img = cap.read()
            if idx % self.read_every == 0:
                img = torch.from_numpy(img).permute(2, 0, 1)
                if self.mode == 'train':
                    img = self.transform['aug'](img)
                else:
                    img = self.transform['normal'](img)
                frames.append(img)

            # depth, height, width, channel => channel, depth, height, width 로 순서 변경
            # (50, 224, 224, 3)             => (3, 50, 224, 224)
            # Conv3D 에 입력으로 넣기 위함
        if self.shape == 'c d h w':
            return torch.stack(frames).permute(1, 0, 2, 3)
        elif self.shape == 'd c h w':
            return torch.stack(frames).permute(0, 1, 2, 3)
    

class CrashDataset(Dataset):
    def __init__(self, dataframe, transform, frame_count=5*10, frame_skip=1, mode='train'):
        super(CrashDataset, self).__init__()
        self.data = dataframe
        self.frame_count = frame_count
        self.mode = mode
        self.read_every = frame_skip
        self.transform = transform
        
    def __getitem__(self, idx):
        video_path = self.data.iloc[idx]['video_path']
        frames = self.load_video(video_path)
        if self.mode == 'train' or self.mode == 'validation':
            label = self.data.iloc[idx]['label']

            # crash
            # 0: no, 1: yes
            if label == 0:
                crash = 0
            else:
                crash = 1
            
            return video_path, frames, ego, weather, timing, label
        else:
            return video_path, frames
        
    def __len__(self):
        return len(self.data)
    
    # 영상 로드
    def load_video(self, video_path):
        frames = []
        cap = cv2.VideoCapture(video_path)
        for idx, _ in enumerate(range(self.frame_count)):
            _, img = cap.read()
            if idx % self.read_every == 0:
                img = torch.from_numpy(img).permute(2, 0, 1)
                if self.mode == 'train':
                    img = self.transform['aug'](img)
                else:
                    img = self.transform['normal'](img)
                frames.append(img)

            # depth, height, width, channel => channel, depth, height, width 로 순서 변경
            # (50, 224, 224, 3)             => (3, 50, 224, 224)
            # Conv3D 에 입력으로 넣기 위함
        return torch.stack(frames).permute(1, 0, 2, 3)
    
    
    
class EWTDatasetFiveCrop(Dataset):
    def __init__(self, dataframe, transform, crop=5, frame_count=5*10, frame_skip=1, mode='train'):
        super(EWTDatasetFiveCrop, self).__init__()
        self.data = dataframe
        self.frame_count = frame_count
        self.mode = mode
        self.read_every = frame_skip
        self.transform = transform
        self.crop = crop
        
    def __getitem__(self, idx):
        video_path = self.data.iloc[idx]['video_path']
        frames = self.load_video(video_path)
        if self.mode == 'train' or self.mode == 'validation':
            label = self.data.iloc[idx]['label']

            # crash
            # 0: no, 1: yes
            if label == 0:
                crash = 0
            else:
                crash = 1
                
            # ego-involve
            # 0: yes, 1: no
            if label >= 1 and label <= 6:
                ego = 0
            elif label >= 7 and label <= 12:
                ego = 1
            else:
                ego = -1
            

            # weather
            # 0: normal, 1: snowy, 2: rainy
            if label in [1, 2, 7, 8]:
                weather = 0
            elif label in [3, 4, 9, 10]:
                weather = 1
            elif label in [5, 6, 11, 12]:
                weather = 2
            else:
                weather = -1

            # timing
            # 0: day, 1: night
            if label in [1, 3, 5, 7, 9, 11]:
                timing = 0
            elif label in [2, 4, 6, 8, 10, 12]:
                timing = 1
            else: 
                timing = -1
            
            return video_path, frames, crash, ego, weather, timing, label
        else:
            return video_path, frames
        
    def __len__(self):
        return len(self.data)
    
    # 영상 로드
    def load_video(self, video_path):
        frames = []
        cap = cv2.VideoCapture(video_path)
        for idx, _ in enumerate(range(self.frame_count)):
            _, img = cap.read()
            if idx % self.read_every == 0:
                img = torch.from_numpy(img).permute(2, 0, 1)
                imgs = self.transform['aug'](img)
                
                for i in range(self.crop):
                    frames.append(imgs[i])

            # depth, height, width, channel => channel, depth, height, width 로 순서 변경
            # (50, 224, 224, 3)             => (3, 50, 224, 224)
            # Conv3D 에 입력으로 넣기 위함
        return torch.stack(frames).permute(1, 0, 2, 3)
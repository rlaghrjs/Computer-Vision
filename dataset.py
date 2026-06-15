import sys
import pandas as pd
import pandas.core.indexes.base as base
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
import pickle
from torchvision import transforms
from sklearn.model_selection import train_test_split


"""
Wafer Dataset 클래스
    1. LSWMD pkl 파일 로드 및 라벨 추출
    2. split 유형('train', 'val', 'test')에 따른 데이터 3분할
    3. 불균형 클래스 비율 유지
    4. 이미지 크기 통일 및 데이터 증강
"""

class WaferDataset(Dataset):
    def __init__(self, pkl_path, img_size=64, split='train'):
        self.img_size = img_size
        self.split = split.lower()
        
        if self.split not in ['train', 'val', 'test']:
            raise ValueError("split 문제 발생.")

        sys.modules["pandas.indexes"] = pd.core.indexes
        sys.modules["pandas.indexes.base"] = base
        
        with open(pkl_path, "rb") as f:
            df = pickle.load(f, encoding="latin1")
        df['failureNum'] = df.failureType.apply(lambda x: len(x))

        labeled_df = df[df['failureNum'] > 0].copy()
        labeled_df['failureType'] = labeled_df.failureType.apply(lambda x: x[0][0])
        
        # 데이터 라벨 매핑
        self.label_mapping = {
            'none': 0, 'Center': 1, 'Donut': 2, 'Edge-Ring': 3, 
            'Edge-Loc': 4, 'Loc': 5, 'Random': 6, 'Scratch': 7, 'Near-full': 8
        }
        labeled_df['target'] = labeled_df['failureType'].map(self.label_mapping)
        
        if self.split == 'train':
            self.data = labeled_df[labeled_df['trianTestLabel'] == 'Training'].reset_index(drop=True)
            
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=(0, 270), interpolation=transforms.InterpolationMode.NEAREST),
                transforms.ToTensor()
            ])
        else:
            original_test_df = labeled_df[labeled_df['trianTestLabel'] == 'Test'].reset_index(drop=True)
            
            val_df, test_df = train_test_split(
                original_test_df, 
                test_size=0.5, 
                random_state=42, 
                stratify=original_test_df['target']
            )
            
            if self.split == 'val':
                self.data = val_df.reset_index(drop=True)
            elif self.split == 'test':
                self.data = test_df.reset_index(drop=True)
                
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor()
            ])
            
        print(f"[{self.split.upper()}] 데이터셋 구조화 완료 : 총 {len(self.data)}장")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        wafer_map = self.data['waferMap'].iloc[idx]
        label = self.data['target'].iloc[idx]
        
        wafer_resized = cv2.resize(wafer_map, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)       
        
        # 3채널 One-Hot 인코딩 적용
        wafer_one_hot = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        for i in range(3):
            wafer_one_hot[:, :, i] = (wafer_resized == i).astype(np.uint8) * 255
            
        if self.transform:
            wafer_tensor = self.transform(wafer_one_hot)
            
        return wafer_tensor, torch.tensor(label, dtype=torch.long)
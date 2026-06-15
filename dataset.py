import sys
import pandas as pd
import pandas.core.indexes.base as base
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
import pickle
from torchvision import transforms


"""
Wafer Dataset 클래스
    1. LSWMD pkl 파일 로드
    2. 라벨 데이터만 추출
    3. Train/Test 분리
    4. 이미지 크기 통일
    5. 데이터 증강(Augmentation)
    6. PyTorch Tensor 형태로 변환
"""

class WaferDataset(Dataset):
    def __init__(self, pkl_path, img_size=64, is_train=True):
        self.img_size = img_size
        self.is_train = is_train

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
        labeled_df['trianTestLabel'] = labeled_df.trianTestLabel.apply(lambda x: x[0][0])


        # 훈련 데이터셋 
        if self.is_train:
            self.data = labeled_df[labeled_df['trianTestLabel'] == 'Training'].reset_index(drop=True)
            
            # 데이터 증강 적용
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=(0, 270), interpolation=transforms.InterpolationMode.NEAREST),
                transforms.ToTensor()
            ])
        # 테스트 데이터셋
        else:
            self.data = labeled_df[labeled_df['trianTestLabel'] == 'Test'].reset_index(drop=True)
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor()
            ])
            
        print(f"[{'Train' if is_train else 'Test'}] 데이터셋 : 총 {len(self.data)}장")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        wafer_map = self.data['waferMap'].iloc[idx]
        label = self.data['target'].iloc[idx]
        
        # 1. 크기 조정 (보간법은 무조건 NEAREST 유지)
        wafer_resized = cv2.resize(wafer_map, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)       
        
        # 2. numpy array를 0, 1, 2 정수형 Tensor로 변환
        wafer_tensor = torch.tensor(wafer_resized, dtype=torch.long)
        
        # 3. One-Hot Encoding 적용 (클래스 3개: 0, 1, 2)
        # 형태 변환: (H, W) -> (H, W, 3) -> (3, H, W)
        wafer_one_hot = torch.nn.functional.one_hot(wafer_tensor, num_classes=3).permute(2, 0, 1).float()
        
        # 4. 데이터 증강 적용 (is_train일 때만)
        # One-Hot 텐서에 바로 적용할 수 있도록 torchvision 0.8+ 지원 기능 활용
        if self.is_train:
            # 주의: RandomRotation 적용 시 빈 공간은 0번 채널(배경)로 채워지도록 수정 필요
            wafer_one_hot = transforms.RandomHorizontalFlip(p=0.5)(wafer_one_hot)
            wafer_one_hot = transforms.RandomVerticalFlip(p=0.5)(wafer_one_hot)
            # 회전 시 밖으로 나가는 여백(fill)은 배경 채널(1,0,0)로 채움
            wafer_one_hot = transforms.functional.rotate(wafer_one_hot, angle=torch.randint(0, 270, (1,)).item(), 
                                                        interpolation=transforms.InterpolationMode.NEAREST, 
                                                        fill=[1.0, 0.0, 0.0])
            
        return wafer_one_hot, torch.tensor(label, dtype=torch.long)
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import time
from torch.utils.data import DataLoader, WeightedRandomSampler, TensorDataset, ConcatDataset
from dataset import WaferDataset
from cnn_models import SimpleWaferCNN, DeepWaferCNN

# ---------------------------------------------------------
# 1. 학습 및 검증 핵심 함수 (기존 로직 유지)
# ---------------------------------------------------------
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
    epoch_loss = running_loss / total
    epoch_acc = (correct / total) * 100
    return epoch_loss, epoch_acc

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    epoch_loss = running_loss / total
    epoch_acc = (correct / total) * 100
    return epoch_loss, epoch_acc

# ---------------------------------------------------------
# 2. 메인 하이브리드 학습 파이프라인
# ---------------------------------------------------------
def main():
    # 하이퍼파라미터 설정
    EPOCHS = 30
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    IMG_SIZE = 64
    SYNTHETIC_DATA_PATH = 'synthetic_wafer_data.pt'
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"장치 설정: {device}")
    
    # 1) 순수 진짜 데이터셋 로드 (Train / Test)
    print("진짜 데이터셋 로드 중...")
    real_train_dataset = WaferDataset(pkl_path='LSWMD.pkl', img_size=IMG_SIZE, is_train=True)
    val_dataset = WaferDataset(pkl_path='LSWMD.pkl', img_size=IMG_SIZE, is_train=False)
    
    real_train_labels = real_train_dataset.data['target'].values
    
    # 2) WGAN이 생성한 가짜 데이터셋 로드 및 병합
    if os.path.exists(SYNTHETIC_DATA_PATH):
        print(f"🔥 가짜 데이터셋 발견 ({SYNTHETIC_DATA_PATH})! 병합을 시작합니다.")
        synthetic_data = torch.load(SYNTHETIC_DATA_PATH, map_location='cpu')
        
        fake_images = synthetic_data['images']
        fake_labels = synthetic_data['labels']
        
        # 텐서 데이터를 파이토치 Dataset 형태로 변환
        synthetic_dataset = TensorDataset(fake_images, fake_labels)
        
        # 💡 핵심: 진짜 데이터셋과 가짜 데이터셋을 하나로 묶음
        train_dataset = ConcatDataset([real_train_dataset, synthetic_dataset])
        
        # 샘플러 설정을 위해 전체 라벨 배열 통합
        combined_labels = np.concatenate([real_train_labels, fake_labels.numpy()])
        print(f" -> 병합 완료! 총 훈련 데이터 수: {len(train_dataset)}장 (진짜: {len(real_train_dataset)}장, 가짜: {len(synthetic_dataset)}장)")
    else:
        print("⚠️ 주의: 가짜 데이터셋 파일을 찾지 못했습니다. 순수 진짜 데이터로만 학습을 진행합니다.")
        train_dataset = real_train_dataset
        combined_labels = real_train_labels

    # 3) 통합 데이터셋 기반 가중치 샘플러 초기화
    # 아무리 데이터를 증강했어도 'none'(정상) 클래스가 훨씬 많을 수 있으므로 최종 밸런스를 잡아줍니다.
    class_counts = np.bincount(combined_labels)
    class_weights = 1.0 / (class_counts + 1e-5)
    sample_weights = [class_weights[label] for label in combined_labels]
    
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    
    # DataLoader 세팅
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    # 4) 모델 및 손실함수 정의 (입력 채널 3으로 바뀐 DeepWaferCNN 사용)
    model = DeepWaferCNN(num_classes=9).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    best_val_acc = 0.0
    
    print("\n🚀 [황금 밸런스 데이터셋] 결함 분류 모델 학습 시작")
    for epoch in range(EPOCHS):
        start_time = time.time()
        
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        epoch_time = time.time() - start_time
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] ({epoch_time:.1f}s) "
              f"| Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% "
              f"| Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        # 최고 성능 모델 저장
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_wafer_model.pth')
            print(f"🌟 최고 검증 정확도 갱신 ({best_val_acc:.2f}%) -> 'best_wafer_model.pth' 저장 완료")

if __name__ == "__main__":
    main()
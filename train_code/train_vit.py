import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler
from dataset import WaferDataset
import time
from vit_models import WaferViT

# 💡 [근본 해결 1] 데이터 불균형 개정을 위한 Focal Loss 클래스 직접 정의
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        if alpha is not None:
            self.register_buffer('alpha', alpha)
        else:
            self.register_buffer('alpha', None)

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # 모델이 맞출 확률
        
        # 확률이 낮을수록(틀릴수록) 거대한 패널티를 부여하는 공식
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.alpha is not None:
            alpha_weight = self.alpha.gather(0, targets.view(-1))
            focal_loss = alpha_weight * focal_loss
            
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# 학습 함수
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

# 검증 함수
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

def main():
    # 💡 [변경] ViT 충분한 학습을 위해 에폭 상향 및 안정적인 Learning Rate 설정
    EPOCHS = 150          
    BATCH_SIZE = 64
    LEARNING_RATE = 5e-4  # ViT 안정성을 위해 0.001에서 0.0005로 하향 조정
    PKL_PATH = 'LSWMD.pkl'
    
    # 조기 종료(Early Stopping) 설정 파라미터
    PATIENCE = 15
    patience_counter = 0
    best_val_loss = float('inf')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_dataset = WaferDataset(pkl_path=PKL_PATH, img_size=64, is_train=True)
    val_dataset = WaferDataset(pkl_path=PKL_PATH, img_size=64, is_train=False)
    
    train_labels = np.array([label for _, label in train_dataset])
    
    class_counts = np.bincount(train_labels)
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[label] for label in train_labels]
    
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    model = WaferViT(num_classes=9).to(device)
    
    # 💡 [근본 해결 2] CrossEntropyLoss 제거 후 데이터 맞춤형 Focal Loss 장착
    alpha_weights = class_weights / class_weights.sum()
    alpha_tensor = torch.FloatTensor(alpha_weights).to(device)
    criterion = FocalLoss(alpha=None, gamma=2.0)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 💡 [근본 해결 3] 유연한 최적화를 위한 코사인 스케줄러 추가
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    
    best_val_acc = 0.0
    
    print(f"\n고도화된 WaferViT + Focal Loss 학습 시작 (최대 {EPOCHS} 에폭).")
    for epoch in range(EPOCHS):
        start_time = time.time()
        
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # 에폭 끝날 때마다 학습률 업데이트
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        epoch_time = time.time() - start_time
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] ({epoch_time:.1f}s) | LR: {current_lr:.6f} "
              f"| Train Loss: {train_loss:.4f} TA: {train_acc:.2f}% "
              f"| Val Loss: {val_loss:.4f} VA: {val_acc:.2f}%")
        
        # 최고 정확도 모델 저장
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_wafer_vit_model.pth')
            print(f"   🔥 최고 정확도 갱신 모델 저장 완료 ({best_val_acc:.2f}%)")
            
        # 💡 [근본 해결 4] 무의미한 반복을 막는 조기 종료(Early Stopping) 로직
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0  # 개선되었으니 카운터 초기화
        else:
            patience_counter += 1  # 손실이 줄어들지 않으면 카운트 증가
            
        if patience_counter >= PATIENCE:
            print(f"\n🛑 조기 종료 트리거: 검증 손실(Val Loss)이 {PATIENCE}에폭 동안 개선되지 않아 학습을 자동 중단합니다.")
            break

    print(f"\n최종 학습 완료. 최고 검증 정확도: {best_val_acc:.2f}%")

if __name__ == "__main__":
    main()
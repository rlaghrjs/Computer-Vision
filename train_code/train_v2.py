import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler
from dataset import WaferDataset
from cnn_models import SimpleWaferCNN, DeepWaferCNN
import time

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
    EPOCHS = 15
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    PKL_PATH = 'LSWMD.pkl'
    
    # 장치 선택
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_dataset = WaferDataset(pkl_path=PKL_PATH, img_size=64, is_train=True)
    val_dataset = WaferDataset(pkl_path=PKL_PATH, img_size=64, is_train=False)
    
    train_labels = np.array([label for _, label in train_dataset])
    
    class_counts = np.bincount(train_labels)
    class_weights = 1.0 / class_counts
    
    sample_weights = [class_weights[label] for label in train_labels]
    
    # 가중치 샘플러
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    model = DeepWaferCNN(num_classes=9).to(device)
    
    criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    best_val_acc = 0.0
    
    print("\n결함 분류 모델 학습 시작.")
    for epoch in range(EPOCHS):
        start_time = time.time()
        
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        epoch_time = time.time() - start_time
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] ({epoch_time:.1f}s) "
              f"| Train Loss: {train_loss:.4f} TA: {train_acc:.2f}% "
              f"| Val Loss: {val_loss:.4f} VA: {val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_wafer_model.pth')
            print(f"모델 저장 완료 ({best_val_acc:.2f}%)")

if __name__ == "__main__":
    main()
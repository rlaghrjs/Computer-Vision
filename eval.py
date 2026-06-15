import torch
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader
from dataset import WaferDataset
from cnn_models import SimpleWaferCNN, DeepWaferCNN

def evaluate_and_measure_time():
    PKL_PATH = 'LSWMD.pkl'
    BATCH_SIZE = 1 
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    test_dataset = WaferDataset(pkl_path=PKL_PATH, img_size=64, is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = DeepWaferCNN(num_classes=9).to(device)
    model.load_state_dict(torch.load('best_wafer_model.pth', map_location=device))
    model.eval()
    
    all_preds = []
    all_labels = []
    inference_times = []
    
    print("\n최종 테스트 및 추론 속도 측정")
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
                
            start_time = time.perf_counter()

            outputs = model(images)
            _, predicted = outputs.max(1)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
                
            end_time = time.perf_counter()
            
            inference_times.append((end_time - start_time) * 1000)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    avg_inference_time = np.mean(inference_times)
    fps = 1000 / avg_inference_time
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    
 
    print(f"▪️ 이미지 1장당 평균 추론 시간 : {avg_inference_time:.2f} ms")
    print(f"▪️ 초당 처리 가능한 프레임 수 (FPS) : {fps:.1f} FPS")
    print(f"▪️ 데이터 불균형 극복 지표 (Macro F1) : {macro_f1:.4f}")
    
    target_names = ['none', 'Center', 'Donut', 'Edge-Ring', 'Edge-Loc', 'Loc', 'Random', 'Scratch', 'Near-full']
    print("\n상세 평가지표:")
    print(classification_report(all_labels, all_preds, target_names=target_names, zero_division=0))
    

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.title('Wafer Map Defect Confusion Matrix')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    plt.savefig('confusion_matrix.png')
    print("Confusion Matrix 그래프 저장.")

if __name__ == "__main__":
    evaluate_and_measure_time()
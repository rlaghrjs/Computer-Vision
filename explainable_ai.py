import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt
from dataset import WaferDataset  # 기존 Dataset 클래스 경로
from cnn_models import DeepWaferCNN   # 제공해주신 모델 클래스 경로

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.features = None
        
        # 피처맵과 그래디언트를 추출하기 위한 훅(Hook) 등록
        self.target_layer.register_forward_hook(self.save_feature)
        self.target_layer.register_full_backward_hook(self.save_gradient)
        
    def save_feature(self, module, input, output):
        self.features = output.detach()
        
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
        
    def generate_heatmap(self, input_tensor, class_idx):
        self.model.eval()
        
        # 1. 순전파 진행 및 특성맵 추출 (Flatten 처리를 가정한 전방 전파)
        # 만약 forward 함수 내부에서 view나 flatten을 수행하는 구조를 명시적으로 재현
        x = self.model.features(input_tensor)
        x_flat = x.view(x.size(0), -1)
        output = self.model.classifier(x_flat)
        
        # 2. 역전파를 통한 그래디언트 계산
        self.model.zero_grad()
        target_score = output[0][class_idx]
        target_score.backward()
        
        # 3. 채널별 가중치 계산 (Global Average Pooling)
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        
        # 4. 가중치와 피처맵 선형 결합 후 ReLU 적용
        cam = torch.sum(weights * self.features, dim=1).squeeze(0)
        cam = np.maximum(cam.cpu().numpy(), 0)
        
        # 5. 0~1 사이 정규화 및 원본 크기(64x64)로 확장
        if cam.max() != 0:
            cam = cam / cam.max()
        cam = cv2.resize(cam, (input_tensor.shape[2], input_tensor.shape[3]))
        return cam

def plot_gradcam_result(original_img, heatmap, true_label, pred_label):
    target_names = ['none', 'Center', 'Donut', 'Edge-Ring', 'Edge-Loc', 'Loc', 'Random', 'Scratch', 'Near-full']
    
    # 텐서를 이미지 규격으로 변환
    img_np = original_img.squeeze().cpu().numpy()
    img_normalized = ((img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8) * 255).astype(np.uint8)
    
    # 컬러 히트맵 생성
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    
    # 원본 이미지 위에 오버레이
    img_3ch = cv2.cvtColor(img_normalized, cv2.COLOR_GRAY2RGB)
    overlayed_img = cv2.addWeighted(img_3ch, 0.6, heatmap_color, 0.4, 0)
    
    # 결과 시각화 출력 (3분할 뷰)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(img_normalized, cmap='gray')
    axes[0].set_title(f"Original Wafer Map\n(True: {target_names[true_label]})", fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(heatmap, cmap='jet')
    axes[1].set_title("Grad-CAM Activation", fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    axes[2].imshow(overlayed_img)
    axes[2].set_title(f"Decision Overlay\n(Pred: {target_names[pred_label]})", fontsize=12, fontweight='bold')
    axes[2].axis('off')
    
    plt.suptitle("Explainable AI (XAI) - Wafer Defect Localization", fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    output_filename = 'wafer_xai_result.png'
    plt.savefig(output_filename, dpi=300)
    print(f"✅ 히트맵 시각화 성공! '{output_filename}' 파일로 저장되었습니다.")
    plt.show()

def main():
    # 1. 모델 및 가중치 로드
    model = DeepWaferCNN(num_classes=9)
    model.load_state_dict(torch.load('best_wafer_model.pth', map_location='cpu'))
    
    target_layer = model.features[14]
    grad_cam = GradCAM(model, target_layer)
    
    # 2. 데이터셋 로드
    dataset = WaferDataset(pkl_path='LSWMD.pkl', img_size=64, is_train=False)
    
    # 🎯 [여기만 수정하세요!] 보고 싶은 불량 클래스 번호 지정
    # 0:none, 1:Center, 2:Donut, 3:Edge-Ring, 4:Edge-Loc, 5:Loc, 6:Random, 7:Scratch, 8:Near-full
    WANTED_CLASS = 8  # 예: 7번을 적으면 스크래치 불량만 찾아옵니다!
    
    target_names = ['none', 'Center', 'Donut', 'Edge-Ring', 'Edge-Loc', 'Loc', 'Random', 'Scratch', 'Near-full']
    print(f"🔎 테스트 데이터셋에서 [{target_names[WANTED_CLASS]}] 불량 샘플을 탐색 중입니다...")
    
    input_tensor = None
    true_label = None
    
    # 데이터셋을 앞에서부터 순회하며 내가 원하는 불량 라벨을 가진 첫 번째 데이터 포착
    for idx in range(len(dataset)):
        img, label = dataset[idx]
        
        if hasattr(label, 'item'):
            label = label.item()
        label = int(label)
        
        if label == WANTED_CLASS:
            input_tensor = img.unsqueeze(0)  # 배치 차원 추가 (1, 1, 64, 64)
            true_label = label
            print(f"✨ 성공! 데이터셋의 {idx}번째 인덱스에서 [{target_names[WANTED_CLASS]}] 데이터를 찾았습니다.")
            break
            
    if input_tensor is None:
        print(f"❌ 아쉽게도 데이터셋에서 {target_names[WANTED_CLASS]} 클래스를 찾지 못했습니다.")
        return
    
    # 3. 모델 예측 실행
    with torch.no_grad():
        x = model.features(input_tensor)
        x_flat = x.view(x.size(0), -1)
        pred_output = model.classifier(x_flat)
        pred_class = torch.argmax(pred_output, dim=1).item()
    
    # 4. 히트맵 생성 및 출력
    heatmap = grad_cam.generate_heatmap(input_tensor, pred_class)
    plot_gradcam_result(input_tensor, heatmap, true_label, pred_class)

if __name__ == "__main__":
    main()
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from dataset import WaferDataset  # 기존에 정의된 Dataset 클래스 호출

def visualize_samples():
    PKL_PATH = 'LSWMD.pkl'
    IMG_SIZE = 64
    
    target_names = ['none (Normal)', 'Center', 'Donut', 'Edge-Ring', 
                    'Edge-Loc', 'Loc', 'Random', 'Scratch', 'Near-full']
    
    print("📦 시각화를 위해 데이터셋을 로드하는 중입니다...")
    dataset = WaferDataset(pkl_path=PKL_PATH, img_size=IMG_SIZE, is_train=False)
    
    found_samples = {}
    
    print("🔍 데이터셋에서 클래스별 샘플을 탐색 중입니다...")
    # 데이터셋을 순회하며 각 클래스별 첫 번째 등장 샘플 확보
    for img, label in dataset:
        # ★ [수정] 파이토치 텐서나 넘파이 형태의 라벨을 순수 파이썬 정수(int)로 변환
        if hasattr(label, 'item'):
            label = label.item()
        label = int(label)
        
        if label not in found_samples:
            found_samples[label] = img
            print(f"-> {target_names[label]} 클래스 샘플 확보 완료!")
            
        if len(found_samples) == 9:  # 9개 클래스를 다 찾으면 루프 종료
            break

    # 시각화 그리드 설정 (3행 3열)
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    axes = axes.ravel()
    
    print("\n🎨 클래스별 Wafer Map 이미지를 그리는 중...")
    for i in range(9):
        if i in found_samples:
            img = found_samples[i]
            
            # 파이토치 텐서(Ch, H, W)를 넘파이(H, W) 형태로 변환
            if isinstance(img, torch.Tensor):
                img = img.squeeze().numpy()
            elif isinstance(img, np.ndarray) and img.ndim == 3:
                img = img.squeeze()
                
            # 이미지 출력 (웨이퍼 패턴이 잘 보이도록 색상 테마 적용)
            axes[i].imshow(img, cmap='twilight_r') 
            axes[i].set_title(f"Class {i}: {target_names[i]}", fontsize=12, fontweight='bold')
        else:
            axes[i].text(0.5, 0.5, 'No Sample', ha='center', va='center', color='red')
            
        axes[i].axis('off')  # 축 눈금 제거
        
    plt.suptitle("WM-811K Wafer Map Defect Patterns", fontsize=16, fontweight='bold', y=0.95)
    plt.tight_layout()
    
    output_filename = 'wafer_defect_samples.png'
    plt.savefig(output_filename, dpi=300)
    print(f"\n✅ 시각화 완료! 결과 이미지가 '{output_filename}'으로 저장되었습니다.")
    plt.show()

if __name__ == "__main__":
    visualize_samples()
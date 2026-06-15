import torch
import numpy as np
import os
from dataset import WaferDataset
from cwgan_gp import Generator

def main():
    # ---------------------------------------------------------
    # 하이퍼파라미터 및 설정
    # ---------------------------------------------------------
    LATENT_DIM = 100            # 1단계와 반드시 동일해야 합니다.
    IMG_SIZE = 64               # 1단계와 반드시 동일해야 합니다.
    NUM_CLASSES = 9
    BATCH_SIZE = 64             # 생성할 때 사용할 배치 크기
    
    # 💡 핵심 설정: 모든 불량 클래스가 최소한 이 개수만큼은 되도록 맞춥니다.
    # 'none'(정상)을 제외한 나머지 1~8번 결함 클래스들의 목표 수량입니다.
    TARGET_COUNT_PER_CLASS = 7000 
    
    # 1단계에서 학습 완료된 가중치 파일 경로
    MODEL_PATH = 'wgan_generator_epoch_200.pth' 
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"장치 설정: {device}")
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 에러: 학습된 WGAN 가중치 파일({MODEL_PATH})을 찾을 수 없습니다. 1단계를 먼저 완료해주세요.")
        return

    # ---------------------------------------------------------
    # 데이터셋 로드 및 현재 클래스별 분포 확인
    # ---------------------------------------------------------
    print("현재 진짜 데이터셋의 결함 분포 확인 중...")
    train_dataset = WaferDataset(pkl_path='LSWMD.pkl', img_size=IMG_SIZE, is_train=True)
    
    # dataset.py 내부의 pandas DataFrame에서 바로 라벨 추출 (속도 최적화)
    real_labels = train_dataset.data['target'].values
    class_counts = np.bincount(real_labels, minlength=NUM_CLASSES)
    
    # 라벨 매핑 텍스트 (출력용)
    label_names = ['none', 'Center', 'Donut', 'Edge-Ring', 'Edge-Loc', 'Loc', 'Random', 'Scratch', 'Near-full']
    
    print("\n[현재 진짜 훈련 데이터 분포]")
    for i, name in enumerate(label_names):
        print(f" - {name} (Class {i}): {class_counts[i]}장")

    # ---------------------------------------------------------
    # Generator 모델 로드 및 가중치 복원
    # ---------------------------------------------------------
    print(f"\n장인 모델({MODEL_PATH}) 로드 중...")
    generator = Generator(latent_dim=LATENT_DIM, num_classes=NUM_CLASSES, img_size=IMG_SIZE).to(device)
    generator.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    generator.eval() # 생성 모드(Evaluation)로 전환 (BatchNorm 등을 고정)

    # 생성된 데이터를 담을 리스트
    fake_images_list = []
    fake_labels_list = []

    print(f"\n🚀 부족한 불량 데이터 생성 시작 (목표: 클래스당 최소 {TARGET_COUNT_PER_CLASS}장)")
    
    # ---------------------------------------------------------
    # 클래스별 가짜 데이터 생성 루프
    # ---------------------------------------------------------
    with torch.no_grad(): # 생성할 때는 그래디언트 계산이 필요 없음
        for class_idx in range(1, NUM_CLASSES): # 0번 'none'(정상)은 제외하고 1~8번 불량만 생성
            current_count = class_counts[class_idx]
            
            # 이미 목표치를 넘은 클래스가 있다면 생성 패스
            if current_count >= TARGET_COUNT_PER_CLASS:
                print(f" -> {label_names[class_idx]} 클래스는 이미 충분합니다. (패스)")
                continue
                
            # 생성해야 할 부족한 수량 계산
            num_to_generate = TARGET_COUNT_PER_CLASS - current_count
            print(f" -> {label_names[class_idx]} (Class {class_idx}): 부족한 {num_to_generate}장 생성 중...", end="")
            
            # GPU 메모리가 터지지 않게 BATCH_SIZE 단위로 쪼개서 생성
            num_batches = int(np.ceil(num_to_generate / BATCH_SIZE))
            
            for b in range(num_batches):
                # 마지막 배치의 남은 수량 처리
                current_batch_size = min(BATCH_SIZE, num_to_generate - (b * BATCH_SIZE))
                
                # 1. 무작위 노이즈 벡터 생성
                z = torch.randn(current_batch_size, LATENT_DIM).to(device)
                
                # 2. 원하는 클래스 라벨 지정 (예: 전부 7번 'Scratch'로 그리라는 명령)
                gen_labels = torch.full((current_batch_size,), class_idx, dtype=torch.long).to(device)
                
                # 3. 가짜 이미지 생성 (Shape: [Batch, 3, 64, 64])
                fake_imgs = generator(z, gen_labels)
                
                # 생성된 텐서를 CPU로 내린 후 리스트에 저장 (GPU 메모리 절약)
                fake_images_list.append(fake_imgs.cpu())
                fake_labels_list.append(gen_labels.cpu())
                
            print(" [완료]")

    # ---------------------------------------------------------
    # 생성된 데이터 병합 및 디스크 저장
    # ---------------------------------------------------------
    if fake_images_list:
        all_fake_images = torch.cat(fake_images_list, dim=0)
        all_fake_labels = torch.cat(fake_labels_list, dim=0)
        
        output_filename = 'synthetic_wafer_data.pt'
        
        # 딕셔너리 형태로 묶어서 한 번에 저장
        torch.save({
            'images': all_fake_images,
            'labels': all_fake_labels
        }, output_filename)
        
        print(f"\n✨ 대성공! 총 {len(all_fake_labels)}장의 고품질 가짜 불량 데이터가 생성되었습니다.")
        print(f"💾 파일 저장 완료: '{output_filename}' (용량 불균형 완벽 해소 완료)")
    else:
        print("\n 생성할 데이터가 없습니다. 모든 클래스가 이미 목표 수량을 충족합니다.")

if __name__ == "__main__":
    main()
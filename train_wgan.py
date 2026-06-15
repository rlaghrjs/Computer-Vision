import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler
import time

# 앞서 수정한 One-Hot Dataset과 CWGAN 모델 임포트
from dataset import WaferDataset
from cwgan_gp import Generator, Critic

# ---------------------------------------------------------
# 1. Gradient Penalty 계산 함수 (WGAN-GP의 핵심)
# ---------------------------------------------------------
def compute_gradient_penalty(critic, real_samples, fake_samples, labels, device):
    """
    진짜 데이터와 가짜 데이터 사이의 무작위 지점(Interpolation)을 생성한 뒤,
    이 지점에서의 기울기(Gradient) 크기가 1에 가까워지도록 패널티를 부여합니다.
    """
    # 무작위 가중치 생성 (0 ~ 1 사이)
    alpha = torch.rand((real_samples.size(0), 1, 1, 1)).to(device)
    
    # 진짜와 가짜를 섞은 보간(Interpolation) 데이터 생성
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    
    # Critic에 보간 데이터 입력
    d_interpolates = critic(interpolates, labels)
    
    # 가짜 출력(가중치) 생성
    fake = torch.autograd.Variable(torch.ones(real_samples.size(0), 1).to(device), requires_grad=False)
    
    # 기울기(Gradient) 계산
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    
    gradients = gradients.view(gradients.size(0), -1)
    
    # 패널티 계산: (기울기의 L2 Norm - 1)^2
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

# ---------------------------------------------------------
# 2. 메인 학습 루프
# ---------------------------------------------------------
def main():
    # 하이퍼파라미터 설정
    EPOCHS = 200            # GAN은 수렴에 시간이 꽤 걸립니다.
    BATCH_SIZE = 64
    LR = 1e-4              # WGAN-GP 권장 학습률
    B1, B2 = 0.0, 0.9      # WGAN-GP 권장 Adam 파라미터
    LATENT_DIM = 100       # 생성기에 들어갈 노이즈(씨앗) 크기
    N_CRITIC = 5           # 생성기를 1번 학습할 때 판별기를 5번 학습 (WGAN 국룰)
    LAMBDA_GP = 10         # Gradient Penalty 가중치
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"장치 설정: {device}")
    
    # 데이터 로더 준비 (작성하신 샘플러 로직 재활용)
    print("데이터셋 로드 중...")
    train_dataset = WaferDataset(pkl_path='LSWMD.pkl', img_size=64, is_train=True)
    
    # GAN 학습 시 'None' 클래스가 너무 많으면 모드 붕괴가 올 수 있으므로 샘플러 필수
    train_labels = np.array([label for _, label in train_dataset])
    class_counts = np.bincount(train_labels)
    class_weights = 1.0 / (class_counts + 1e-5)
    sample_weights = [class_weights[label] for label in train_labels]
    
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=2, drop_last=True)
    
    # 모델 초기화
    generator = Generator(latent_dim=LATENT_DIM, num_classes=9, img_size=64).to(device)
    critic = Critic(num_classes=9, img_size=64).to(device)
    
    # 옵티마이저 설정 (Critic과 Generator 따로)
    optimizer_G = optim.Adam(generator.parameters(), lr=LR, betas=(B1, B2))
    optimizer_C = optim.Adam(critic.parameters(), lr=LR, betas=(B1, B2))
    
    print("\n🚀 WGAN-GP 모델 학습 시작")
    
    for epoch in range(EPOCHS):
        start_time = time.time()
        
        loss_G_epoch = 0.0
        loss_C_epoch = 0.0
        batches = 0
        
        for i, (imgs, labels) in enumerate(train_loader):
            batches += 1
            real_imgs = imgs.to(device)
            labels = labels.to(device)
            batch_size = real_imgs.size(0)
            
            # ---------------------
            # 판별기(Critic) 학습
            # ---------------------
            optimizer_C.zero_grad()
            
            # 1. 진짜 이미지 평가
            real_validity = critic(real_imgs, labels)
            
            # 2. 가짜 이미지 생성 및 평가
            z = torch.randn(batch_size, LATENT_DIM).to(device) # 무작위 노이즈
            fake_imgs = generator(z, labels)
            fake_validity = critic(fake_imgs.detach(), labels) # detach(): Generator 가중치는 보호
            
            # 3. Gradient Penalty 계산
            gradient_penalty = compute_gradient_penalty(critic, real_imgs.data, fake_imgs.data, labels, device)
            
            # 4. Critic 최종 손실(Loss) 계산
            # 수식: (가짜 점수 평균) - (진짜 점수 평균) + (패널티 * 가중치)
            loss_C = torch.mean(fake_validity) - torch.mean(real_validity) + LAMBDA_GP * gradient_penalty
            
            loss_C.backward()
            optimizer_C.step()
            loss_C_epoch += loss_C.item()
            
            # ---------------------
            # 생성기(Generator) 학습 (N_CRITIC 번당 1번씩 실행)
            # ---------------------
            if i % N_CRITIC == 0:
                optimizer_G.zero_grad()
                
                # 가짜 이미지 다시 생성
                gen_imgs = generator(z, labels)
                
                # 생성기 입장에서는 Critic이 이 이미지를 '진짜'라고 평가하도록 속여야 함
                fake_validity = critic(gen_imgs, labels)
                loss_G = -torch.mean(fake_validity)
                
                loss_G.backward()
                optimizer_G.step()
                loss_G_epoch += loss_G.item()
        
        # 에포크 결과 출력
        epoch_time = time.time() - start_time
        avg_loss_C = loss_C_epoch / batches
        avg_loss_G = loss_G_epoch / (batches / N_CRITIC)
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] ({epoch_time:.1f}s) | Critic Loss: {avg_loss_C:.4f} | Generator Loss: {avg_loss_G:.4f}")
        
        # 10 에포크마다 가중치 저장
        if (epoch + 1) % 10 == 0:
            torch.save(generator.state_dict(), f'wgan_generator_epoch_{epoch+1}.pth')
            print(f"✅ 모델 저장 완료: wgan_generator_epoch_{epoch+1}.pth")

if __name__ == "__main__":
    main()
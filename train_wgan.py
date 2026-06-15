import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler
import time

from dataset import WaferDataset
from cwgan_gp import Generator, Critic


def compute_gradient_penalty(critic, real_samples, fake_samples, labels, device):

    alpha = torch.rand((real_samples.size(0), 1, 1, 1)).to(device)
    
    # 진짜와 가짜를 섞은 데이터 생성
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    
    # Critic에 보간 데이터 입력
    d_interpolates = critic(interpolates, labels)
    
    # 가짜 출력 생성
    fake = torch.autograd.Variable(torch.ones(real_samples.size(0), 1).to(device), requires_grad=False)
    
    # 기울기 계산
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    
    gradients = gradients.view(gradients.size(0), -1)
    
    # 패널티 계산
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def main():
    # 하이퍼파라미터 설정
    EPOCHS = 200           
    BATCH_SIZE = 64
    LR = 1e-4             
    B1, B2 = 0.0, 0.9      
    LATENT_DIM = 100      
    N_CRITIC = 5        
    LAMBDA_GP = 10  
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_dataset = WaferDataset(pkl_path='LSWMD.pkl', img_size=64, is_train=True)
    
    train_labels = np.array([label for _, label in train_dataset])
    class_counts = np.bincount(train_labels)
    class_weights = 1.0 / (class_counts + 1e-5)
    sample_weights = [class_weights[label] for label in train_labels]
    
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=2, drop_last=True)
    
    # 모델 초기화
    generator = Generator(latent_dim=LATENT_DIM, num_classes=9, img_size=64).to(device)
    critic = Critic(num_classes=9, img_size=64).to(device)
    
    # 옵티마이저
    optimizer_G = optim.Adam(generator.parameters(), lr=LR, betas=(B1, B2))
    optimizer_C = optim.Adam(critic.parameters(), lr=LR, betas=(B1, B2))
    
    print("\n모델 학습 시작")
    
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
            
            optimizer_C.zero_grad()

            real_validity = critic(real_imgs, labels)
            
            z = torch.randn(batch_size, LATENT_DIM).to(device) 
            fake_imgs = generator(z, labels)
            fake_validity = critic(fake_imgs.detach(), labels) 

            # Gradient Penalty 계산
            gradient_penalty = compute_gradient_penalty(critic, real_imgs.data, fake_imgs.data, labels, device)
            
            # 4. Critic 최종 손실 계산
            loss_C = torch.mean(fake_validity) - torch.mean(real_validity) + LAMBDA_GP * gradient_penalty
            
            loss_C.backward()
            optimizer_C.step()
            loss_C_epoch += loss_C.item()
            

            if i % N_CRITIC == 0:
                optimizer_G.zero_grad()
                
                # 가짜 이미지 다시 생성
                gen_imgs = generator(z, labels)
                
                fake_validity = critic(gen_imgs, labels)
                loss_G = -torch.mean(fake_validity)
                
                loss_G.backward()
                optimizer_G.step()
                loss_G_epoch += loss_G.item()
        
        epoch_time = time.time() - start_time
        avg_loss_C = loss_C_epoch / batches
        avg_loss_G = loss_G_epoch / (batches / N_CRITIC)
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] ({epoch_time:.1f}s) | Critic Loss: {avg_loss_C:.4f} | Generator Loss: {avg_loss_G:.4f}")
        
        # 10 에포크마다 가중치 저장
        if (epoch + 1) % 10 == 0:
            torch.save(generator.state_dict(), f'wgan_generator_epoch_{epoch+1}.pth')
            print(f"모델 저장 완료: wgan_generator_epoch_{epoch+1}.pth")

if __name__ == "__main__":
    main()
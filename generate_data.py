import torch
import numpy as np
import os
from dataset import WaferDataset
from cwgan_gp import Generator

def main():

    LATENT_DIM = 100  
    IMG_SIZE = 64               
    NUM_CLASSES = 9
    BATCH_SIZE = 64  
    
    # 모든 불량 클래스가 최소 갯수 맞추기
    TARGET_COUNT_PER_CLASS = 7000 
    
    # 가중치 파일 경로
    MODEL_PATH = 'wgan_generator_epoch_200.pth' 
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if not os.path.exists(MODEL_PATH):
        print(f"WGAN 가중치 파일({MODEL_PATH})을 찾을 수 없음")
        return

    train_dataset = WaferDataset(pkl_path='LSWMD.pkl', img_size=IMG_SIZE, is_train=True)
    
    real_labels = train_dataset.data['target'].values
    class_counts = np.bincount(real_labels, minlength=NUM_CLASSES)
    
    # 라벨 매핑 텍스트
    label_names = ['none', 'Center', 'Donut', 'Edge-Ring', 'Edge-Loc', 'Loc', 'Random', 'Scratch', 'Near-full']
    
    print("\n현재 데이터 분포 상황")
    for i, name in enumerate(label_names):
        print(f" - {name} (Class {i}): {class_counts[i]}장")


    generator = Generator(latent_dim=LATENT_DIM, num_classes=NUM_CLASSES, img_size=IMG_SIZE).to(device)
    generator.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    generator.eval()

    fake_images_list = []
    fake_labels_list = []

    print(f"\n데이터 생성 시작 {TARGET_COUNT_PER_CLASS}장)")
    

    with torch.no_grad(): 
        # 정상을 제외한 1~8번 불량만 생성
        for class_idx in range(1, NUM_CLASSES):
            current_count = class_counts[class_idx]
            
            # 이미 목표치를 넘은 클래스가 있다면 생성 패스
            if current_count >= TARGET_COUNT_PER_CLASS:            
                continue
                
            # 생성해야 할 부족한 수량 계산
            num_to_generate = TARGET_COUNT_PER_CLASS - current_count
            
            num_batches = int(np.ceil(num_to_generate / BATCH_SIZE))
            
            for b in range(num_batches):
                current_batch_size = min(BATCH_SIZE, num_to_generate - (b * BATCH_SIZE))               
    
                z = torch.randn(current_batch_size, LATENT_DIM).to(device)
                
                gen_labels = torch.full((current_batch_size,), class_idx, dtype=torch.long).to(device)
                
                fake_imgs = generator(z, gen_labels)
                
                fake_images_list.append(fake_imgs.cpu())
                fake_labels_list.append(gen_labels.cpu())
                
            print("완료")


    if fake_images_list:
        all_fake_images = torch.cat(fake_images_list, dim=0)
        all_fake_labels = torch.cat(fake_labels_list, dim=0)
        
        output_filename = 'synthetic_wafer_data.pt'
        
        torch.save({
            'images': all_fake_images,
            'labels': all_fake_labels
        }, output_filename)
        
        print(f"파일 저장: '{output_filename}")
    else:
        print("\n 생성할 데이터가 없음")

if __name__ == "__main__":
    main()
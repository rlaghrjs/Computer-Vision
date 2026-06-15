import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbedding(nn.Module):
    """이미지를 조각(Patch) 내어 특징 벡터(Embedding)로 바꾸는 레이어"""
    def __init__(self, in_channels=1, patch_size=16, embed_dim=256):
        super().__init__()
        self.patch_size = patch_size
        # [핵심] CNN 필터(Conv2d)의 스트라이드와 커널을 patch_size로 두어 조각냄
        self.projection = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (Batch, 1, 64, 64) -> projection: (Batch, embed_dim, 4, 4)
        x = self.projection(x)
        # Flatten: (Batch, embed_dim, 16) -> Transpose: (Batch, 16, embed_dim)
        x = x.flatten(2).transpose(1, 2) 
        return x

class WaferViT(nn.Module):
    def __init__(self, num_classes=9, embed_dim=256, depth=6, num_heads=8, patch_size=16, dropout=0.3):
        super().__init__()
        
        # 1. 이미지를 패치 임베딩으로 변환
        self.patch_embedding = PatchEmbedding(1, patch_size, embed_dim)
        
        # 2. [핵심] [CLS] 토큰: 전체 이미지를 대표하는 학습 가능한 특징 토큰
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # 3. 위치 임베딩(Position Embedding): 조각들의 위치 정보를 추가
        # (1 + 4*4 = 17)
        self.pos_embedding = nn.Parameter(torch.zeros(1, 17, embed_dim))
        
        # 4. 트랜스포머 인코더 블록 (숲을 보는 뇌 영역)
        # depth개의 층을 쌓아 전역적 관계를 학습
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=512, dropout=dropout, activation='gelu', batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        # 5. 최종 분류기 (MLP Head)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        batch_size = x.size(0)
        
        # [A] 패치 추출: (Batch, 16, embed_dim)
        x = self.patch_embedding(x)
        
        # [B] CLS 토큰 복사 및 결합
        # (Batch, 1, embed_dim)
        cls_token = self.cls_token.expand(batch_size, -1, -1) 
        # (Batch, 17, embed_dim)
        x = torch.cat((cls_token, x), dim=1) 
        
        # [C] 위치 정보 추가
        x = x + self.pos_embedding
        
        # [D] 트랜스포머 인코더 통과 (전역 관계 학습)
        x = self.transformer_encoder(x)
        
        # [E] 최종 판단: [CLS] 토큰의 최종 특징만 추출 (Batch, embed_dim)
        cls_output = x[:, 0]
        
        # [F] 분류
        out = self.classifier(cls_output)
        return out
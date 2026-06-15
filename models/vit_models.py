import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=1, patch_size=16, embed_dim=256):
        super().__init__()
        self.patch_size = patch_size
        self.projection = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.projection(x)
        x = x.flatten(2).transpose(1, 2) 
        return x

class WaferViT(nn.Module):
    def __init__(self, num_classes=9, embed_dim=256, depth=6, num_heads=8, patch_size=16, dropout=0.3):
        super().__init__()
        
        self.patch_embedding = PatchEmbedding(1, patch_size, embed_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.pos_embedding = nn.Parameter(torch.zeros(1, 17, embed_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=512, dropout=dropout, activation='gelu', batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

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
        
        x = self.patch_embedding(x)
        
        cls_token = self.cls_token.expand(batch_size, -1, -1) 
        x = torch.cat((cls_token, x), dim=1) 
        
        x = x + self.pos_embedding
        
        x = self.transformer_encoder(x)
        
        cls_output = x[:, 0]
        
        out = self.classifier(cls_output)
        return out
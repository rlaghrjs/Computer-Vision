import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, latent_dim=100, num_classes=9, img_size=64):
        super(Generator, self).__init__()
        self.img_size = img_size
        self.label_emb = nn.Embedding(num_classes, num_classes)
        
        self.init_size = img_size // 4
        self.l1 = nn.Sequential(nn.Linear(latent_dim + num_classes, 128 * self.init_size ** 2))
        
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 3, 3, stride=1, padding=1),
            nn.Softmax(dim=1) 
        )

    def forward(self, noise, labels):
        c = self.label_emb(labels)
        gen_input = torch.cat((noise, c), -1)
        out = self.l1(gen_input)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

class Critic(nn.Module):
    def __init__(self, num_classes=9, img_size=64):
        super(Critic, self).__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)
        
        self.model = nn.Sequential(
            nn.Conv2d(3 + num_classes, 64, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.adv_layer = nn.Linear(512 * (img_size // 16) ** 2, 1)

    def forward(self, img, labels):
        c = self.label_emb(labels)
        c = c.view(c.size(0), c.size(1), 1, 1)
        c = c.repeat(1, 1, img.size(2), img.size(3))
        
        crit_input = torch.cat((img, c), 1)
        out = self.model(crit_input)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        return validity
import torch
import torch.nn as nn
import torchvision.models as models

class RGBBackbone(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()

        # EfficientNet_B0 (stable, fast)
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.encoder = m.features

        # project to embedding
        self.proj = nn.Sequential(
            nn.Conv2d(1280, embed_dim, kernel_size=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):  
        feat = self.encoder(x)          # (B,1280,H/32,W/32)
        out = self.proj(feat)           # (B,256,H/32,W/32)
        return out

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone_rgb import RGBBackbone
from .backbone_depth import DepthBackbone

class LiteFusionModel(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()

        # Backbones
        self.rgb = RGBBackbone(embed_dim)
        self.depth = DepthBackbone(embed_dim)

        # Fusion head (concat + MLP)
        self.fusion = nn.Sequential(
            nn.Conv2d(embed_dim * 2, 256, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        # Regression heads
        self.head_aqi = nn.Linear(256, 1)
        self.head_pm25 = nn.Linear(256, 1)
        self.head_pm10 = nn.Linear(256, 1)

    def forward(self, rgb, depth):
        fr = self.rgb(rgb)                      # (B,256,h,w)
        fd = self.depth(depth)                  # (B,256,h,w) resized? No mismatch.

        # resize depth to match rgb spatial dims
        if fr.shape[-1] != fd.shape[-1]:
            fd = F.interpolate(fd, size=fr.shape[-2:], mode="bilinear")

        fused = torch.cat([fr, fd], dim=1)      # (B,512,h,w)
        fused = self.fusion(fused)              # (B,256,1,1)
        fused = fused.flatten(1)                # (B,256)

        return {
            "aqi":  self.head_aqi(fused).squeeze(1),
            "pm25": self.head_pm25(fused).squeeze(1),
            "pm10": self.head_pm10(fused).squeeze(1),
        }

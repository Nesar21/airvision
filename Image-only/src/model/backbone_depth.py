import torch
import torch.nn as nn

class DepthBackbone(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, embed_dim, 3, stride=2, padding=1),
            nn.ReLU(),
        )

    def forward(self, d):
        return self.net(d)              # (B,256,H/16,W/16)

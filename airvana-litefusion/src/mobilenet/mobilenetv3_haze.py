import os
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small

class MobileNetV3Haze(nn.Module):
    """
    Simple 2-class haze/smog classifier on top of MobileNetV3-small.
    """
    def __init__(self, num_classes: int = 2, backbone_ckpt: str | None = None, device: str = "cpu"):
        super().__init__()
        self.backbone = mobilenet_v3_small(weights=None)

        # Optionally load your regression backbone weights (features only)
        if backbone_ckpt and os.path.exists(backbone_ckpt):
            state = torch.load(backbone_ckpt, map_location=device)
            feat_state = {k: v for k, v in state.items() if k.startswith("features.")}
            missing, unexpected = self.backbone.load_state_dict(feat_state, strict=False)
            print("[MobileNetV3Haze] Loaded backbone from", backbone_ckpt)
            print("  missing keys:", missing)
            print("  unexpected keys:", unexpected)

        in_features = self.backbone.classifier[3].in_features
        # Replace classifier head
        self.backbone.classifier[3] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)

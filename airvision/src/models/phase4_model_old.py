# src/models/phase4_model.py

from dataclasses import dataclass
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


# ---------------------------------------------------------
# CONFIG STRUCT
# ---------------------------------------------------------


@dataclass
class AQIModelConfig:
    img_backbone_out_dim: int = 1024          # final backbone embedding dim
    engineered_dim: int = 12                  # number of engineered features (Phase 2)
    engineered_proj_dim: int = 64             # projection dim for engineered features
    weather_dim: int = 0                      # set >0 in Phase 7 for fusion
    haze_num_classes: int = 6                 # AQI_Class levels / haze bins
    dropout_p: float = 0.3                    # MC Dropout rate
    use_imagenet_weights: bool = True         # EffNet-B0 pretrained weights


# ---------------------------------------------------------
# BACKBONE: EfficientNet-B0 → 1024D
# ---------------------------------------------------------


class EfficientNetBackbone(nn.Module):
    """
    EfficientNet-B0 backbone that outputs a fixed-size embedding.
    - Uses torchvision.efficientnet_b0
    - Strips classifier
    - Adds Linear projection to img_backbone_out_dim (default 1024)
    """

    def __init__(self, out_dim: int, use_imagenet_weights: bool = True):
        super().__init__()
        if use_imagenet_weights:
            weights = EfficientNet_B0_Weights.IMAGENET1K_V1
        else:
            weights = None

        backbone = efficientnet_b0(weights=weights)

        # backbone.features produces spatial feature maps
        self.feature_extractor = backbone.features
        # classifier[1] is the final Linear; its in_features is the pooled feature dim
        feat_dim = backbone.classifier[1].in_features

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.proj = nn.Linear(feat_dim, out_dim)

        nn.init.kaiming_normal_(self.proj.weight, nonlinearity="linear")
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 3, H, W]
        returns: [B, out_dim]
        """
        feats = self.feature_extractor(x)      # [B, C, H', W']
        feats = self.pool(feats)               # [B, C, 1, 1]
        feats = self.flatten(feats)            # [B, C]
        out = self.proj(feats)                 # [B, out_dim]
        return out


# ---------------------------------------------------------
# MAIN MODEL: IMAGE + ENGINEERED FEATURES (+ OPTIONAL WEATHER)
# HEADS: AQI, log-variance, haze classification, visibility
# ---------------------------------------------------------


class AQIModel(nn.Module):
    """
    Phase 4 core model:
    - Backbone: EfficientNet-B0 → 1024D embedding
    - Engineered features: 12D → Linear(64) + LayerNorm + ReLU
    - Combined embedding: concat(backbone, engineered_proj) → 1088D
      (fusion with weather will append extra dims in Phase 7)
    - Dropout on combined embedding (MC Dropout ready)
    - Heads:
        * AQI regression head: scalar
        * log-variance head (heteroscedastic): scalar, clamped in loss
        * haze classification head: C-way logits
        * visibility regression head: scalar
    """

    def __init__(self, cfg: AQIModelConfig):
        super().__init__()
        self.cfg = cfg

        # Backbone
        self.backbone = EfficientNetBackbone(
            out_dim=cfg.img_backbone_out_dim,
            use_imagenet_weights=cfg.use_imagenet_weights,
        )

        # Engineered features projection: Dense(64, small_init=Kaiming*0.5) → LN → ReLU
        self.engineered_proj = nn.Sequential(
            nn.Linear(cfg.engineered_dim, cfg.engineered_proj_dim),
            nn.LayerNorm(cfg.engineered_proj_dim),
            nn.ReLU(inplace=True),
        )

        # Small-init for engineered projection
        lin: nn.Linear = self.engineered_proj[0]
        nn.init.kaiming_normal_(lin.weight, nonlinearity="relu")
        lin.weight.data *= 0.5
        nn.init.zeros_(lin.bias)

        # Combined dim: backbone (1024) + engineered (64) + optional weather
        combined_dim = cfg.img_backbone_out_dim + cfg.engineered_proj_dim + cfg.weather_dim

        self.dropout = nn.Dropout(p=cfg.dropout_p)

        # Heads
        self.aqi_head = nn.Linear(combined_dim, 1)                     # mean
        self.logvar_head = nn.Linear(combined_dim, 1)                  # log-variance
        self.haze_head = nn.Linear(combined_dim, cfg.haze_num_classes) # haze / AQI class
        self.vis_head = nn.Linear(combined_dim, 1)                     # visibility

        # Standard init for heads
        for head in [self.aqi_head, self.logvar_head, self.haze_head, self.vis_head]:
            nn.init.kaiming_normal_(head.weight, nonlinearity="linear")
            nn.init.zeros_(head.bias)

        self.float()  # enforce float32

    def forward(
        self,
        images: torch.Tensor,
        engineered_feats: torch.Tensor,
        weather_feats: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        images:          [B, 3, H, W]
        engineered_feats:[B, engineered_dim]  (already Z-scored via feature_stats.json)
        weather_feats:   [B, weather_dim] or None (for fusion in Phase 7)

        returns dict:
            {
                "embedding":       [B, combined_dim],
                "aqi_mean":        [B, 1],
                "aqi_logvar":      [B, 1],
                "haze_logits":     [B, C],
                "visibility":      [B, 1],
            }
        """
        # Backbone embedding
        img_emb = self.backbone(images)  # [B, 1024]

        # Engineered features
        eng_emb = self.engineered_proj(engineered_feats)  # [B, 64]

        # Fusion (image-only or image+weather)
        if weather_feats is not None:
            combined = torch.cat([img_emb, eng_emb, weather_feats], dim=1)
        else:
            combined = torch.cat([img_emb, eng_emb], dim=1)  # [B, 1088 for weather_dim=0]

        combined = self.dropout(combined)

        # Heads
        aqi_mean = self.aqi_head(combined)      # [B, 1]
        aqi_logvar = self.logvar_head(combined) # [B, 1], clamp later in loss: [-10, 10]
        haze_logits = self.haze_head(combined)  # [B, C]
        visibility = self.vis_head(combined)    # [B, 1]

        return {
            "embedding": combined,
            "aqi_mean": aqi_mean,
            "aqi_logvar": aqi_logvar,
            "haze_logits": haze_logits,
            "visibility": visibility,
        }


# ---------------------------------------------------------
# OPTIONAL: SIMPLE SELF-TEST
# ---------------------------------------------------------


def _self_test() -> None:
    """
    Quick sanity check:
    - Runs a forward pass with dummy tensors
    - Prints output shapes
    This is for developer validation only (not used in training).
    """
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    cfg = AQIModelConfig(
        img_backbone_out_dim=1024,
        engineered_dim=12,
        engineered_proj_dim=64,
        weather_dim=0,
        haze_num_classes=6,
        dropout_p=0.3,
        use_imagenet_weights=False,  # keep False for quick local tests
    )

    model = AQIModel(cfg).to(device)
    model.eval()

    B = 2
    images = torch.randn(B, 3, 224, 224, device=device, dtype=torch.float32)
    engineered_feats = torch.randn(B, cfg.engineered_dim, device=device, dtype=torch.float32)

    with torch.no_grad():
        out = model(images, engineered_feats)

    print("embedding:", out["embedding"].shape)
    print("aqi_mean:", out["aqi_mean"].shape)
    print("aqi_logvar:", out["aqi_logvar"].shape)
    print("haze_logits:", out["haze_logits"].shape)
    print("visibility:", out["visibility"].shape)


if __name__ == "__main__":
    _self_test()

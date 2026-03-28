import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
except Exception:
    timm = None

# Expected backbone output dim (EffNet-B0 = 1024)
BACKBONE_OUT_DIM = 1024
ENGINEERED_PROJ_OUT = 64  # 524D → 64D


class Phase4Model(nn.Module):
    """
    Ablation-ready Phase-4 model.

    Output contract (MANDATORY for Phase-6):
        {
            "aqi_mean": Tensor[B],
            "aqi_logvar": Tensor[B] or None,
            "haze_logits": Tensor[B, C] or None,
            "vis_logits": Tensor[B] or None,
        }
    """

    def __init__(
        self,
        use_phase2_embedding: bool = True,
        use_haze_head: bool = True,
        use_vis_head: bool = True,
        use_uncertainty: bool = True,
        backbone_name: str = "efficientnet_b0",
        pretrained_backbone: bool = True,
    ):
        super().__init__()

        self.use_phase2_embedding = bool(use_phase2_embedding)
        self.use_haze_head = bool(use_haze_head)
        self.use_vis_head = bool(use_vis_head)
        self.use_uncertainty = bool(use_uncertainty)

        # ------------------------------
        # Backbone
        # ------------------------------
        if timm is None:
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(32, BACKBONE_OUT_DIM),
            )
            self.backbone_out_dim = BACKBONE_OUT_DIM
        else:
            m = timm.create_model(backbone_name, pretrained=pretrained_backbone)

            if hasattr(m, "num_features") and m.num_features:
                backbone_out = m.num_features
            elif hasattr(m, "classifier") and hasattr(m.classifier, "in_features"):
                backbone_out = m.classifier.in_features
            else:
                backbone_out = BACKBONE_OUT_DIM

            if hasattr(m, "classifier"):
                m.classifier = nn.Identity()
            if hasattr(m, "fc"):
                m.fc = nn.Identity()

            self.backbone = m
            self.backbone_out_dim = backbone_out

        if self.backbone_out_dim != BACKBONE_OUT_DIM:
            print(
                f"[WARN] Backbone output dim {self.backbone_out_dim} differs from expected "
                f"{BACKBONE_OUT_DIM}. Using {self.backbone_out_dim} for heads."
            )

        # ------------------------------
        # Engineered projection
        # ------------------------------
        if self.use_phase2_embedding:
            self.engineered_proj = nn.Sequential(
                nn.Linear(524, 64),
                nn.LayerNorm(64),
                nn.ReLU(inplace=True),
                nn.Linear(64, ENGINEERED_PROJ_OUT),
            )
            engineered_out_dim = ENGINEERED_PROJ_OUT
        else:
            self.engineered_proj = None
            engineered_out_dim = 0

        self.combined_dim = self.backbone_out_dim + engineered_out_dim

        # ------------------------------
        # Heads
        # ------------------------------
        self.aqi_head = nn.Sequential(
            nn.Linear(self.combined_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )

        self.logvar_head = (
            nn.Sequential(
                nn.Linear(self.combined_dim, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 1),
            )
            if self.use_uncertainty
            else None
        )

        self.haze_head = (
            nn.Sequential(
                nn.Linear(self.combined_dim, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 3),
            )
            if self.use_haze_head
            else None
        )

        self.vis_head = (
            nn.Sequential(
                nn.Linear(self.combined_dim, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 1),
            )
            if self.use_vis_head
            else None
        )

        self.dropout = nn.Dropout(p=0.3)

        self.LOGVAR_MIN = -10.0
        self.LOGVAR_MAX = 10.0

    # --------------------------
    # Feature extraction
    # --------------------------
    def extract_image_features(self, images):
        feat = self.backbone(images)
        if isinstance(feat, (tuple, list)):
            feat = feat[0]
        if feat.dim() == 4:
            feat = F.adaptive_avg_pool2d(feat, 1).reshape(feat.size(0), -1)
        return feat.view(feat.size(0), -1)

    # --------------------------
    # Forward (PHASE-6 SAFE)
    # --------------------------
    def forward(self, images, engineered=None):
        img_feat = self.extract_image_features(images)

        if self.use_phase2_embedding:
            if engineered is None:
                raise ValueError(
                    "Engineered features required when use_phase2_embedding=True"
                )
            eng = engineered.view(engineered.size(0), -1)
            eng_proj = self.engineered_proj(eng)
            combined = torch.cat([img_feat, eng_proj], dim=1)
        else:
            combined = img_feat

        combined = self.dropout(combined)

        # --------------------------
        # OUTPUT CONTRACT (CRITICAL)
        # --------------------------
        out = {}

        # AQI mean (ALWAYS present)
        out["aqi_mean"] = self.aqi_head(combined).squeeze(-1)

        # Uncertainty (optional)
        if self.logvar_head is not None:
            raw = self.logvar_head(combined).squeeze(-1)
            out["aqi_logvar"] = torch.clamp(
                raw, self.LOGVAR_MIN, self.LOGVAR_MAX
            )
        else:
            out["aqi_logvar"] = None

        # Haze (optional)
        out["haze_logits"] = (
            self.haze_head(combined) if self.haze_head is not None else None
        )

        # Visibility (optional)
        out["vis_logits"] = (
            self.vis_head(combined).squeeze(-1)
            if self.vis_head is not None
            else None
        )

        return out

    def forward_image_only(self, images):
        if self.use_phase2_embedding:
            raise RuntimeError(
                "forward_image_only requires use_phase2_embedding=False"
            )
        return self.forward(images, engineered=None)

    def summary(self):
        return {
            "use_phase2_embedding": self.use_phase2_embedding,
            "use_haze_head": self.use_haze_head,
            "use_vis_head": self.use_vis_head,
            "use_uncertainty": self.use_uncertainty,
            "backbone_out_dim": self.backbone_out_dim,
            "combined_dim": self.combined_dim,
        }

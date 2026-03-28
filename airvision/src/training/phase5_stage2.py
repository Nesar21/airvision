# src/training/phase5_stage2.py
"""
Phase 5 — Stage 2: Soft AQI Pretraining (PM25Vision ONLY)

Stage Goal:
    - Continue from Stage 1 haze-pretrained backbone.
    - Train soft AQI regression using heteroscedastic loss.
    - Normalize AQI to [0,1] before training.
    - Output global backbone for Stage 3 (IND_NEP fine-tuning).
"""

from pathlib import Path
from typing import Dict, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision.transforms import v2 as T

from src.utils.env_utils import init_env
from src.utils.manifest import log_run
from src.models.phase4_model_old import AQIModel, AQIModelConfig


# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------

class Stage2Config:
    project_root = Path(".")
    pm25_root = project_root / "pm25vision"

    train_meta = pm25_root / "train" / "metadata.csv"
    train_images = pm25_root / "train" / "images"
    val_meta = pm25_root / "test" / "metadata.csv"
    val_images = pm25_root / "test" / "images"

    batch_size = 16
    num_workers = 4
    max_epochs = 32
    early_stop_patience = 8

    backbone_lr = 1e-5
    head_lr = 1e-3
    weight_decay = 1e-5
    grad_clip = 1.0

    base_label_conf = 0.5

    ckpt_dir = project_root / "checkpoints"
    ckpt_stage1 = ckpt_dir / "backbone_haze.pth"
    ckpt_stage2 = ckpt_dir / "backbone_global.pth"


# ---------------------------------------------------------
# PM2.5 → EPA AQI
# ---------------------------------------------------------

def pm25_to_aqi_epa(pm25: torch.Tensor) -> torch.Tensor:
    bp = [
        (0.0, 12.0, 0.0, 50.0),
        (12.1, 35.4, 51.0, 100.0),
        (35.5, 55.4, 101.0, 150.0),
        (55.5, 150.4, 151.0, 200.0),
        (150.5, 250.4, 201.0, 300.0),
        (250.5, 350.4, 301.0, 400.0),
        (350.5, 500.4, 401.0, 500.0),
    ]

    aqi = torch.zeros_like(pm25)
    for C_low, C_high, I_low, I_high in bp:
        mask = (pm25 >= C_low) & (pm25 <= C_high)
        if mask.any():
            aqi[mask] = (I_high - I_low) / (C_high - C_low) * (pm25[mask] - C_low) + I_low

    return torch.clamp(aqi, 0.0, 500.0)


# ---------------------------------------------------------
# DATASET
# ---------------------------------------------------------

class PM25VisionSoftAQIDataset(Dataset):
    def __init__(self, meta_csv: Path, image_root: Path, base_label_conf: float):
        import pandas as pd
        self.df = pd.read_csv(meta_csv)
        self.image_root = image_root
        self.base_label_conf = base_label_conf

        if "pm25" not in self.df.columns:
            raise ValueError(f"'pm25' missing in {meta_csv}")

        self.transform = T.Compose([
            T.ConvertImageDtype(torch.float32),
            T.Resize(256),
            T.CenterCrop(224),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        self.engineered_dim = 12

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        img = read_image(str(self.image_root / row["filename"]))
        img = self.transform(img)

        pm = torch.tensor(float(row["pm25"]), dtype=torch.float32)
        soft_aqi = pm25_to_aqi_epa(pm.unsqueeze(0)).squeeze(0)
        soft_aqi = soft_aqi / 500.0   # NORMALIZE

        engineered = torch.zeros(self.engineered_dim, dtype=torch.float32)
        label_conf = torch.tensor(self.base_label_conf, dtype=torch.float32)

        return {
            "images": img,
            "engineered": engineered,
            "soft_aqi": soft_aqi,
            "label_conf": label_conf,
        }


# ---------------------------------------------------------
# DATALOADERS
# ---------------------------------------------------------

def build_dataloaders(cfg: Stage2Config):
    train_ds = PM25VisionSoftAQIDataset(cfg.train_meta, cfg.train_images, cfg.base_label_conf)
    val_ds = PM25VisionSoftAQIDataset(cfg.val_meta, cfg.val_images, cfg.base_label_conf)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True
    )

    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True
    )

    return train_loader, val_loader


# ---------------------------------------------------------
# MODEL + OPTIM
# ---------------------------------------------------------

def build_model_and_optim(cfg: Stage2Config, device):
    model_cfg = AQIModelConfig(
        img_backbone_out_dim=1024,
        engineered_dim=12,
        engineered_proj_dim=64,
        weather_dim=0,
        haze_num_classes=6,
        dropout_p=0.3,
        use_imagenet_weights=False,
    )

    model = AQIModel(model_cfg).to(device)

    ckpt = torch.load(cfg.ckpt_stage1, map_location=device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    print(f"[Stage2] Loaded Stage1 → {cfg.ckpt_stage1}")

    backbone_params = list(model.backbone.parameters())

    head_params = (
        list(model.engineered_proj.parameters()) +
        list(model.aqi_head.parameters()) +
        list(model.logvar_head.parameters()) +
        list(model.haze_head.parameters()) +
        list(model.vis_head.parameters())
    )

    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": cfg.backbone_lr},
            {"params": head_params, "lr": cfg.head_lr},
        ],
        weight_decay=cfg.weight_decay,
    )

    return model, optimizer, model_cfg


# ---------------------------------------------------------
# LOSS
# ---------------------------------------------------------

def hetero_loss(mean, logvar, target, conf):
    logvar = torch.clamp(logvar, -10, 10)
    var_inv = torch.exp(-logvar)
    mse = (target - mean) ** 2
    nll = 0.5 * (var_inv * mse + logvar)
    return (conf * nll).mean()


# ---------------------------------------------------------
# EPOCH LOOP
# ---------------------------------------------------------

def run_epoch(model, loader, optimizer, device, cfg, train=True):
    model.train() if train else model.eval()

    tot_loss = 0.0
    tot_mae = 0.0
    tot = 0

    for batch in loader:
        images = batch["images"].to(device)
        engineered = batch["engineered"].to(device)
        soft_aqi = batch["soft_aqi"].to(device)
        conf = batch["label_conf"].to(device)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            out = model(images, engineered)

            mean = out["aqi_mean"].squeeze(-1) / 500.0
            logvar = out["aqi_logvar"].squeeze(-1)

            loss = hetero_loss(mean, logvar, soft_aqi, conf)

            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimizer.step()

        B = images.size(0)
        tot_loss += loss.item() * B
        tot_mae += torch.abs(mean.detach() - soft_aqi).sum().item()
        tot += B

    return {"loss": tot_loss / tot, "mae": tot_mae / tot}


# ---------------------------------------------------------
# TRAINER
# ---------------------------------------------------------

def train_stage2():
    device = init_env(False)
    cfg = Stage2Config()
    cfg.ckpt_dir.mkdir(exist_ok=True)

    train_loader, val_loader = build_dataloaders(cfg)
    model, optimizer, model_cfg = build_model_and_optim(cfg, device)

    best_loss = float("inf")
    best_state = None
    patience = 0

    for epoch in range(1, cfg.max_epochs + 1):
        tr = run_epoch(model, train_loader, optimizer, device, cfg, train=True)
        va = run_epoch(model, val_loader, optimizer, device, cfg, train=False)

        print(
            f"[Stage2][Epoch {epoch:03d}] "
            f"train_loss={tr['loss']:.4f} train_mae={tr['mae']:.3f} "
            f"val_loss={va['loss']:.4f} val_mae={va['mae']:.3f}"
        )

        if va["loss"] < best_loss:
            best_loss = va["loss"]
            best_state = {
                "model_state": model.state_dict(),
                "model_cfg": model_cfg.__dict__,
                "epoch": epoch,
                "val_loss": va["loss"],
                "val_mae": va["mae"],
            }
            patience = 0
        else:
            patience += 1
            if patience >= cfg.early_stop_patience:
                print("[Stage2] Early stopping triggered.")
                break

    torch.save(best_state, cfg.ckpt_stage2)
    print(f"[Stage2] Saved global backbone → {cfg.ckpt_stage2}")

    log_run(
        phase="phase5_stage2",
        stage="soft_aqi_pretrain",
        description="Stage 2 normalized AQI heteroscedastic training",
        config={"batch_size": cfg.batch_size, "lr": cfg.backbone_lr},
        metrics={
            "best_val_loss": best_loss,
            "best_val_mae": best_state["val_mae"],
        },
    )


if __name__ == "__main__":
    train_stage2()

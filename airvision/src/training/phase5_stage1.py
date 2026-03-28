# src/training/phase5_stage1.py

"""
Phase 5 — Stage 1: Haze Pretraining (PM25Vision ONLY)
"""

import os
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision.transforms import v2 as T

from src.utils.env_utils import init_env
from src.utils.manifest import log_run
from src.models.phase4_model_old import AQIModel, AQIModelConfig


class Stage1Config:
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

    lambda_haze = 1.0
    lambda_vis = 0.0
    lambda_soft_stage1 = 0.0

    haze_bins = ["0–50", "51–100", "101–150", "151–200", "201–300", "301–600"]

    ckpt_dir = project_root / "checkpoints"
    ckpt_path = ckpt_dir / "backbone_haze.pth"


class PM25VisionHazeDataset(Dataset):
    def __init__(self, meta_csv: Path, image_root: Path, haze_bins):
        import pandas as pd

        self.df = pd.read_csv(meta_csv)
        self.image_root = image_root
        self.haze_map = {b: i for i, b in enumerate(haze_bins)}

        unknown = set(self.df["pm25_bin"].unique()) - set(self.haze_map.keys())
        if unknown:
            raise ValueError(f"Unknown pm25_bin values: {unknown}")

        self.transform = T.Compose(
            [
                T.ConvertImageDtype(torch.float32),
                T.Resize(256),
                T.CenterCrop(224),
                T.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        self.engineered_dim = 12

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = read_image(str(self.image_root / row["filename"]))
        img = self.transform(img)

        haze_label = self.haze_map[row["pm25_bin"]]
        engineered_feats = torch.zeros(self.engineered_dim, dtype=torch.float32)

        return {
            "images": img,
            "engineered": engineered_feats,
            "haze_label": torch.tensor(haze_label, dtype=torch.long),
        }


def build_dataloaders(cfg: Stage1Config):
    train_ds = PM25VisionHazeDataset(cfg.train_meta, cfg.train_images, cfg.haze_bins)
    val_ds = PM25VisionHazeDataset(cfg.val_meta, cfg.val_images, cfg.haze_bins)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True
    )
    return train_loader, val_loader


def build_model_and_optim(cfg: Stage1Config, device):
    model_cfg = AQIModelConfig(
        img_backbone_out_dim=1024,
        engineered_dim=12,
        engineered_proj_dim=64,
        weather_dim=0,
        haze_num_classes=len(cfg.haze_bins),
        dropout_p=0.3,
        use_imagenet_weights=True,
    )

    model = AQIModel(model_cfg).to(device)

    backbone_params = list(model.backbone.parameters())
    head_params = (
        list(model.engineered_proj.parameters())
        + list(model.aqi_head.parameters())
        + list(model.logvar_head.parameters())
        + list(model.haze_head.parameters())
        + list(model.vis_head.parameters())
    )

    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": cfg.backbone_lr},
            {"params": head_params, "lr": cfg.head_lr},
        ],
        weight_decay=cfg.weight_decay,
    )

    return model, optimizer, model_cfg


def run_epoch(model, loader, optimizer, device, cfg, train=True):
    model.train() if train else model.eval()

    ce_loss = nn.CrossEntropyLoss()
    total_loss = total_haze = total_correct = total_samples = 0

    for batch in loader:
        images = batch["images"].to(device)
        engineered = batch["engineered"].to(device)
        haze_label = batch["haze_label"].to(device)

        if train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            out = model(images, engineered)
            haze_logits = out["haze_logits"]

            L_haze = ce_loss(haze_logits, haze_label)
            L_vis = torch.tensor(0.0, device=device)

            loss = cfg.lambda_haze * L_haze + cfg.lambda_vis * L_vis

            if train:
                loss.backward()

                ### FIX 1 — CORRECT GRADIENT CLIP CALL
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

                optimizer.step()

        B = images.size(0)
        total_loss += loss.item() * B
        total_haze += L_haze.item() * B

        preds = haze_logits.argmax(1)
        total_correct += (preds == haze_label).sum().item()
        total_samples += B

    return {
        "loss": total_loss / total_samples,
        "haze_loss": total_haze / total_samples,
        "haze_acc": total_correct / total_samples,
        "num_samples": total_samples,
    }


def train_stage1():
    device = init_env()
    cfg = Stage1Config()
    cfg.ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader = build_dataloaders(cfg)
    model, optimizer, model_cfg = build_model_and_optim(cfg, device)

    best_loss = float("inf")
    best_state = None
    patience = 0

    for epoch in range(1, cfg.max_epochs + 1):
        tr = run_epoch(model, train_loader, optimizer, device, cfg, train=True)
        va = run_epoch(model, val_loader, optimizer, device, cfg, train=False)

        print(
            f"[Stage1][Epoch {epoch:03d}] "
            f"train_loss={tr['loss']:.4f} acc={tr['haze_acc']:.4f} "
            f"val_loss={va['loss']:.4f} acc={va['haze_acc']:.4f}"
        )

        if va["loss"] < best_loss:
            best_loss = va["loss"]
            patience = 0

            best_state = {
                "model_state": model.state_dict(),

                ### FIX 2 — SAFE SERIALIZATION OF model_cfg
                "model_cfg": model_cfg.__dict__,

                "epoch": epoch,
                "val_loss": va["loss"],
                "val_haze_acc": va["haze_acc"],
            }

        else:
            patience += 1
            if patience >= cfg.early_stop_patience:
                print("[Stage1] Early stopping triggered.")
                break

    torch.save(best_state, cfg.ckpt_path)
    print(f"[Stage1] Saved checkpoint → {cfg.ckpt_path}")

    log_run(
        phase="phase5_stage1",
        stage="haze_pretrain_pm25vision",
        description="Stage 1 haze pretraining on PM25Vision",
        config={
            "batch_size": cfg.batch_size,
            "epochs": cfg.max_epochs,
            "backbone_lr": cfg.backbone_lr,
            "head_lr": cfg.head_lr,
        },
        metrics={
            "best_val_loss": best_state["val_loss"],
            "best_val_haze_acc": best_state["val_haze_acc"],
        },
    )


if __name__ == "__main__":
    train_stage1()

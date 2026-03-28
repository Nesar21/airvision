# litefusion_kfold_train.py
# Run:
#   python litefusion_kfold_train.py --fold 0 --config cfg.yaml
#
# This trains ANY selected fold:
#    splits/fold0/train.csv
#    splits/fold0/val.csv
#
# For k=4 folds, run fold 0,1,2,3 separately.

import os
import json
import time
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

# --------------------------------------------
# DEVICE (MPS → CUDA → CPU)
# --------------------------------------------
DEVICE = (
    "mps"
    if torch.backends.mps.is_available()
    else ("cuda" if torch.cuda.is_available() else "cpu")
)

# --------------------------------------------
# CONFIG DEFAULTS
# --------------------------------------------
DEFAULT_CFG = {
    "project_root": ".",
    "splits_dir": "splits",
    "fold": 0,

    "img_size": 256,
    "batch_size": 32,
    "epochs": 50,
    "lr": 1e-4,
    "weight_decay": 1e-2,
    "device": DEVICE,
    "num_workers": 4,

    "save_dir": "outputs",
    "backbone": "mobilenetv3_large_100",
    "pretrained": True,

    "w_aqi": 1.0,
    "w_pm25": 0.7,
    "w_pm10": 0.7
}

# --------------------------------------------
# DATASET
# --------------------------------------------
class AQIMultiDataset(Dataset):
    def __init__(self, csv_path, img_size=256):
        self.df = pd.read_csv(csv_path)
        self.img_size = img_size

        self.to_tensor = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.CenterCrop((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.df)

    def _load_image(self, p):
        try:
            img = Image.open(p).convert("RGB")
        except:
            img = Image.new("RGB", (self.img_size, self.img_size), (128,128,128))
        return self.to_tensor(img)

    def _load_depth(self, p):
        if not isinstance(p, str) or p == "":
            return None
        try:
            arr = np.load(p)
            if arr.ndim != 2:
                return None
            norm = (arr - arr.min()) / (arr.ptp() + 1e-8)
            img = Image.fromarray((norm*255).astype("uint8")).convert("L")
            img = img.resize((self.img_size, self.img_size))
            return transforms.ToTensor()(img)
        except:
            return None

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        img = self._load_image(r["image_path"])
        depth = self._load_depth(r.get("depth_path",""))
        if depth is None:
            depth = torch.zeros(1, self.img_size, self.img_size)

        # numeric labels
        pm25 = float(r["pm25"]) if not pd.isna(r["pm25"]) else np.nan
        pm10 = float(r["pm10"]) if not pd.isna(r["pm10"]) else np.nan
        aqi  = float(r["aqi"])  if not pd.isna(r["aqi"])  else np.nan

        return {
            "image": img,
            "depth": depth,
            "aqi": aqi,
            "pm25": pm25,
            "pm10": pm10,
            "has_aqi": 0 if pd.isna(aqi) else 1,
            "has_pm25": 0 if pd.isna(pm25) else 1,
            "has_pm10": 0 if pd.isna(pm10) else 1
        }


def collate(batch):
    B = len(batch)
    imgs  = torch.stack([b["image"] for b in batch])
    depths= torch.stack([b["depth"] for b in batch])

    aqi  = torch.tensor([b["aqi"] for b in batch], dtype=torch.float32)
    pm25 = torch.tensor([b["pm25"] for b in batch], dtype=torch.float32)
    pm10 = torch.tensor([b["pm10"] for b in batch], dtype=torch.float32)

    has_aqi  = torch.tensor([b["has_aqi"] for b in batch], dtype=torch.bool)
    has_pm25 = torch.tensor([b["has_pm25"] for b in batch], dtype=torch.bool)
    has_pm10 = torch.tensor([b["has_pm10"] for b in batch], dtype=torch.bool)

    return {
        "image": imgs,
        "depth": depths,
        "aqi": aqi,
        "pm25": pm25,
        "pm10": pm10,
        "has_aqi": has_aqi,
        "has_pm25": has_pm25,
        "has_pm10": has_pm10
    }

# --------------------------------------------
# MODEL
# --------------------------------------------
class NumericEncoder(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim),
            nn.LayerNorm(out_dim)
        )

    def forward(self, x):
        return self.mlp(x)


class LiteFusion(nn.Module):
    def __init__(self, backbone_name, pretrained, img_size=256):
        super().__init__()

        # Create backbone
        self.backbone = timm.create_model(
            backbone_name, pretrained=pretrained,
            num_classes=0, global_pool="avg"
        )

        feat_dim = self.backbone.num_features

        self.img_proj = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.ReLU(),
            nn.LayerNorm(512)
        )

        self.depth_conv = nn.Sequential(
            nn.Conv2d(1,16,3,stride=2,padding=1), nn.ReLU(),
            nn.Conv2d(16,32,3,stride=2,padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32,128),
            nn.ReLU(),
            nn.LayerNorm(128)
        )

        self.numeric = NumericEncoder(out_dim=128)

        self.fusion = nn.Sequential(
            nn.Linear(512+128+128, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.LayerNorm(256)
        )

        self.head_aqi  = nn.Linear(256,1)
        self.head_pm25 = nn.Linear(256,1)
        self.head_pm10 = nn.Linear(256,1)

    def forward(self, image, depth, num):
        feat = self.backbone(image)
        img_emb = self.img_proj(feat)
        depth_emb = self.depth_conv(depth)
        num_emb = self.numeric(num)

        x = torch.cat([img_emb, depth_emb, num_emb], dim=1)
        z = self.fusion(x)

        return {
            "aqi": self.head_aqi(z).squeeze(1),
            "pm25": self.head_pm25(z).squeeze(1),
            "pm10": self.head_pm10(z).squeeze(1)
        }

# --------------------------------------------
# LOSS
# --------------------------------------------
def masked_loss(preds, targets, mask, cfg):
    L = nn.L1Loss()

    total = 0
    parts = []

    if mask["has_aqi"].any():
        parts.append(cfg["w_aqi"] * L(preds["aqi"][mask["has_aqi"]],
                                      targets["aqi"][mask["has_aqi"]]))
        total += 1

    if mask["has_pm25"].any():
        parts.append(cfg["w_pm25"] * L(preds["pm25"][mask["has_pm25"]],
                                       targets["pm25"][mask["has_pm25"]]))
        total += 1

    if mask["has_pm10"].any():
        parts.append(cfg["w_pm10"] * L(preds["pm10"][mask["has_pm10"]],
                                       targets["pm10"][mask["has_pm10"]]))
        total += 1

    if total == 0:
        return preds["aqi"].sum()*0.0

    return sum(parts)/total

# --------------------------------------------
# TRAIN ONE FOLD
# --------------------------------------------
def train_one_fold(cfg):
    device = torch.device(cfg["device"])

    fold_dir = Path(cfg["splits_dir"]) / f"fold{cfg['fold']}"
    train_csv = fold_dir/"train.csv"
    val_csv   = fold_dir/"val.csv"

    train_loader = DataLoader(
        AQIMultiDataset(train_csv, img_size=cfg["img_size"]),
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        collate_fn=collate
    )

    val_loader = DataLoader(
        AQIMultiDataset(val_csv, img_size=cfg["img_size"]),
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        collate_fn=collate
    )

    model = LiteFusion(cfg["backbone"], cfg["pretrained"], cfg["img_size"]).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=cfg["epochs"])

    save_dir = Path(cfg["save_dir"]) / f"fold{cfg['fold']}"
    save_dir.mkdir(parents=True, exist_ok=True)

    best_mae = 1e9

    for epoch in range(cfg["epochs"]):
        model.train()
        loss_sum = 0
        steps = 0
        t0 = time.time()

        for batch in train_loader:
            img = batch["image"].to(device)
            depth = batch["depth"].to(device)

            pm25 = torch.nan_to_num(batch["pm25"], nan=0.0).to(device)
            pm10 = torch.nan_to_num(batch["pm10"], nan=0.0).to(device)
            aqi0 = torch.zeros(img.shape[0], device=device)

            numeric = torch.stack([pm25, pm10, aqi0], dim=1)

            preds = model(img, depth, numeric)

            targs = {
                "aqi": batch["aqi"].to(device),
                "pm25": batch["pm25"].to(device),
                "pm10": batch["pm10"].to(device)
            }
            mask = {
                "has_aqi": batch["has_aqi"].to(device),
                "has_pm25": batch["has_pm25"].to(device),
                "has_pm10": batch["has_pm10"].to(device)
            }

            loss = masked_loss(preds, targs, mask, cfg)

            optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

            loss_sum += loss.item()
            steps += 1

        sched.step()
        t1 = time.time()

        # validation
        val_mae = evaluate(model, val_loader, device).get("aqi_mae", 999)
        print(f"[FOLD {cfg['fold']}] Epoch {epoch+1}/{cfg['epochs']} "
              f"loss={loss_sum/steps:.4f}  val_mae={val_mae:.4f}  time={t1-t0:.1f}s")

        if val_mae < best_mae:
            best_mae = val_mae
            torch.save(model.state_dict(), save_dir/"best.pth")

    return best_mae


# --------------------------------------------
# EVALUATE
# --------------------------------------------
def evaluate(model, loader, device):
    model.eval()
    ys, ps = [], []

    with torch.no_grad():
        for batch in loader:
            img = batch["image"].to(device)
            depth = batch["depth"].to(device)

            pm25 = torch.nan_to_num(batch["pm25"], nan=0.0).to(device)
            pm10 = torch.nan_to_num(batch["pm10"], nan=0.0).to(device)
            aqi0 = torch.zeros(img.shape[0], device=device)

            num = torch.stack([pm25, pm10, aqi0], dim=1)

            out = model(img, depth, num)

            mask = batch["has_aqi"]
            if mask.any():
                ys.extend(batch["aqi"][mask].tolist())
                ps.extend(out["aqi"][mask].cpu().tolist())

    if len(ys)==0:
        return {"aqi_mae":999}

    mae = mean_absolute_error(ys, ps)
    rmse = mean_squared_error(ys, ps, squared=False)
    r2   = r2_score(ys, ps)

    return {"aqi_mae": mae, "aqi_rmse": rmse, "aqi_r2": r2}


# --------------------------------------------
# MAIN
# --------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()

    cfg = DEFAULT_CFG.copy()
    import yaml
    cfg.update(yaml.safe_load(open(args.config)))
    cfg["fold"] = args.fold

    print(json.dumps(cfg, indent=2))
    score = train_one_fold(cfg)
    print(f"[FINAL] Fold {cfg['fold']} best MAE = {score:.4f}")

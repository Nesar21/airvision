# litefusion_model_and_train.py
# Single-file runnable trainer + model for Image+Depth+Numeric fusion.
# Run with:
#   source ~/venvs/image_aqi/bin/activate
#   python litefusion_model_and_train.py --config cfg.yaml

import os
import math
import time
import json
import argparse
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------------------
# Defaults
# ---------------------------
DEFAULT_CFG = {
    "project_root": ".",
    "master_csv": "data/master_v2.csv",
    "splits_dir": "splits",
    "train_csv": "splits/train.csv",
    "val_csv": "splits/val.csv",
    "test_csv": "splits/test.csv",
    "img_size": 256,
    "batch_size": 16,
    "epochs": 30,
    "lr": 1e-4,
    "weight_decay": 1e-2,
    "device": ("mps" if torch.backends.mps.is_available()
           else ("cuda" if torch.cuda.is_available()
                 else "cpu")),
    "num_workers": 4,
    "save_dir": "outputs",
    "backbone": "mobilenetv3_large_100",
    "pretrained": True,
    "w_aqi": 1.0,
    "w_pm25": 0.7,
    "w_pm10": 0.7,
}

# ---------------------------
# Dataset
# ---------------------------
class AQIMultiDataset(Dataset):
    def __init__(self, csv_path, img_size=256, split="train", augment=False):
        self.df = pd.read_csv(csv_path)
        self.img_size = img_size

        self.to_tensor = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.CenterCrop((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])

    def __len__(self):
        return len(self.df)

    def _load_image(self, path):
        try:
            img = Image.open(path).convert('RGB')
        except Exception:
            img = Image.new('RGB', (self.img_size, self.img_size), (128,128,128))
        return self.to_tensor(img)

    def _load_depth(self, path):
        if not isinstance(path, str) or path == "":
            return None
        try:
            arr = np.load(path)
            if arr.ndim == 2:
                arrn = (arr - arr.min()) / (arr.ptp() + 1e-8)
                img = Image.fromarray((arrn * 255.0).astype("uint8")).convert("L")
                img = img.resize((self.img_size, self.img_size))
                return transforms.ToTensor()(img)
            return None
        except Exception:
            return None

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        image_path = r['image_path']
        depth_path = r.get('depth_path', '')

        img = self._load_image(image_path)
        depth = self._load_depth(depth_path)
        if depth is None:
            depth = torch.zeros(1, self.img_size, self.img_size)
            depth_avail = False
        else:
            depth_avail = True

        pm25 = float(r['pm25']) if not pd.isna(r['pm25']) else np.nan
        pm10 = float(r['pm10']) if not pd.isna(r['pm10']) else np.nan
        aqi  = float(r['aqi'])  if not pd.isna(r['aqi'])  else np.nan

        numeric = {
            'pm25': pm25,
            'pm10': pm10,
            'aqi': aqi,
            'has_pm25': not np.isnan(pm25),
            'has_pm10': not np.isnan(pm10),
            'has_aqi': not np.isnan(aqi),
            'source': r.get('source', 'UNK'),
            'day_night': r.get('day_night', 'Unknown'),
            'depth_avail': depth_avail
        }

        return {
            'image': img,
            'depth': depth,
            'numeric': numeric,
            'image_path': image_path
        }

def collate_batch(batch):
    images = torch.stack([b['image'] for b in batch], dim=0)
    depths = torch.stack([b['depth'] for b in batch], dim=0)

    bs = len(batch)
    aqi  = torch.full((bs,), float("nan"))
    pm25 = torch.full((bs,), float("nan"))
    pm10 = torch.full((bs,), float("nan"))
    has_aqi  = torch.zeros((bs,), dtype=torch.bool)
    has_pm25 = torch.zeros((bs,), dtype=torch.bool)
    has_pm10 = torch.zeros((bs,), dtype=torch.bool)

    for i, b in enumerate(batch):
        n = b['numeric']
        if n['has_aqi']:
            aqi[i] = n['aqi']
            has_aqi[i] = True
        if n['has_pm25']:
            pm25[i] = n['pm25']
            has_pm25[i] = True
        if n['has_pm10']:
            pm10[i] = n['pm10']
            has_pm10[i] = True

    return {
        'image': images,
        'depth': depths,
        'aqi': aqi,
        'pm25': pm25,
        'pm10': pm10,
        'has_aqi': has_aqi,
        'has_pm25': has_pm25,
        'has_pm10': has_pm10,
        'paths': [b['image_path'] for b in batch]
    }

# ---------------------------
# Model
# ---------------------------
class NumericEncoder(nn.Module):
    def __init__(self, hidden=128, out_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x):
        return self.mlp(x)

class LiteFusion(nn.Module):
    def __init__(self, backbone_name='mobilenetv3_large_100', pretrained=True, numeric_dim=128, img_size=256):
        super().__init__()

        try:
            self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0, global_pool='avg')
        except Exception:
            self.backbone = timm.create_model(backbone_name, pretrained=pretrained)

        self.backbone.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, 3, img_size, img_size)
            out = self.backbone(dummy)
            if isinstance(out, (tuple, list)):
                for o in reversed(out):
                    if isinstance(o, torch.Tensor):
                        out_tensor = o
                        break
                else:
                    out_tensor = out[0]
            else:
                out_tensor = out

            if out_tensor.ndim == 4:
                feat_dim = out_tensor.shape[1]
            elif out_tensor.ndim == 2:
                feat_dim = out_tensor.shape[1]
            else:
                feat_dim = getattr(self.backbone, "num_features", None) or 1280

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
            nn.Linear(32,128), nn.ReLU(), nn.LayerNorm(128)
        )

        self.numeric = NumericEncoder(out_dim=numeric_dim)

        self.fusion = nn.Sequential(
            nn.Linear(512 + 128 + numeric_dim, 512),
            nn.ReLU(), nn.LayerNorm(512),
            nn.Linear(512, 256),
            nn.ReLU(), nn.LayerNorm(256)
        )

        self.head_aqi  = nn.Linear(256, 1)
        self.head_pm25 = nn.Linear(256, 1)
        self.head_pm10 = nn.Linear(256, 1)

    def forward(self, image, depth, numeric_input):
        feat = self.backbone(image)
        if isinstance(feat, (tuple, list)):
            for f in feat:
                if isinstance(f, torch.Tensor):
                    feat = f
                    break

        if feat.ndim == 4:
            feat = feat.mean(dim=[2,3])

        img_emb = self.img_proj(feat)
        depth_emb = self.depth_conv(depth)
        num_emb = self.numeric(numeric_input)

        x = torch.cat([img_emb, depth_emb, num_emb], dim=1)
        z = self.fusion(x)

        return {
            'aqi':  self.head_aqi(z).squeeze(1),
            'pm25': self.head_pm25(z).squeeze(1),
            'pm10': self.head_pm10(z).squeeze(1),
        }

# ---------------------------
# Masked loss
# ---------------------------
def masked_multi_task_loss(preds, targets, masks, cfg):
    loss_fn = nn.L1Loss()

    losses = []
    count = 0

    if masks['has_aqi'].any():
        la = loss_fn(preds['aqi'][masks['has_aqi']], targets['aqi'][masks['has_aqi']])
        losses.append(cfg['w_aqi'] * la)
        count += 1

    if masks['has_pm25'].any():
        l25 = loss_fn(preds['pm25'][masks['has_pm25']], targets['pm25'][masks['has_pm25']])
        losses.append(cfg['w_pm25'] * l25)
        count += 1

    if masks['has_pm10'].any():
        l10 = loss_fn(preds['pm10'][masks['has_pm10']], targets['pm10'][masks['has_pm10']])
        losses.append(cfg['w_pm10'] * l10)
        count += 1

    if count == 0:
        return preds['aqi'].sum() * 0.0

    return sum(losses) / count

# ---------------------------
# Evaluation
# ---------------------------
def evaluate(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    pm25_t, pm25_p = [], []
    pm10_t, pm10_p = [] ,[]

    with torch.no_grad():
        for batch in loader:
            img = batch['image'].to(device)
            depth = batch['depth'].to(device)

            pm25_v = torch.nan_to_num(batch['pm25'], nan=0.0).to(device)
            pm10_v = torch.nan_to_num(batch['pm10'], nan=0.0).to(device)
            aqi_bin = torch.zeros(img.shape[0], device=device)

            numeric = torch.stack([pm25_v, pm10_v, aqi_bin], dim=1)
            out = model(img, depth, numeric)

            mask_aqi = batch['has_aqi'].to(device)
            mask25 = batch['has_pm25'].to(device)
            mask10 = batch['has_pm10'].to(device)

            if mask_aqi.any():
                y_true.extend(batch['aqi'][mask_aqi.cpu()].numpy().tolist())
                y_pred.extend(out['aqi'][mask_aqi].cpu().numpy().tolist())

            if mask25.any():
                pm25_t.extend(batch['pm25'][mask25.cpu()].numpy().tolist())
                pm25_p.extend(out['pm25'][mask25].cpu().numpy().tolist())

            if mask10.any():
                pm10_t.extend(batch['pm10'][mask10.cpu()].numpy().tolist())
                pm10_p.extend(out['pm10'][mask10].cpu().numpy().tolist())

    metrics = {}
    if len(y_true) > 0:
        metrics["aqi_mae"]  = mean_absolute_error(y_true, y_pred)
        metrics["aqi_rmse"] = mean_squared_error(y_true, y_pred) ** 0.5
        metrics["aqi_r2"]   = r2_score(y_true, y_pred)

    if len(pm25_t) > 0:
        metrics["pm25_mae"] = mean_absolute_error(pm25_t, pm25_p)

    if len(pm10_t) > 0:
        metrics["pm10_mae"] = mean_absolute_error(pm10_t, pm10_p)

    return metrics

# ---------------------------
# Training
# ---------------------------
def train(cfg):
    device = torch.device(cfg['device'])
    os.makedirs(cfg['save_dir'], exist_ok=True)

    json.dump(cfg, open(Path(cfg['save_dir'])/"cfg.json", "w"), indent=2)

    train_loader = DataLoader(
        AQIMultiDataset(cfg['train_csv'], img_size=cfg['img_size']),
        batch_size=cfg['batch_size'],
        shuffle=True,
        num_workers=cfg['num_workers'],
        collate_fn=collate_batch
    )

    val_loader = DataLoader(
        AQIMultiDataset(cfg['val_csv'], img_size=cfg['img_size']),
        batch_size=cfg['batch_size'],
        shuffle=False,
        num_workers=cfg['num_workers'],
        collate_fn=collate_batch
    )

    model = LiteFusion(backbone_name=cfg['backbone'], pretrained=cfg['pretrained'], img_size=cfg['img_size']).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['epochs'])

    best_val = 1e9

    for epoch in range(cfg['epochs']):
        model.train()
        epoch_loss = 0
        n = 0
        t0 = time.time()

        for batch in train_loader:
            img = batch['image'].to(device)
            depth = batch['depth'].to(device)

            pm25_v = torch.nan_to_num(batch['pm25'], nan=0.0).to(device)
            pm10_v = torch.nan_to_num(batch['pm10'], nan=0.0).to(device)
            aqi_bin = torch.zeros(img.shape[0], device=device)

            numeric = torch.stack([pm25_v, pm10_v, aqi_bin], dim=1)

            out = model(img, depth, numeric)

            targets = {
                'aqi': batch['aqi'].to(device),
                'pm25': batch['pm25'].to(device),
                'pm10': batch['pm10'].to(device)
            }
            masks = {
                'has_aqi': batch['has_aqi'].to(device),
                'has_pm25': batch['has_pm25'].to(device),
                'has_pm10': batch['has_pm10'].to(device)
            }

            loss = masked_multi_task_loss(out, targets, masks, cfg)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n += 1

        scheduler.step()
        t1 = time.time()

        val_m = evaluate(model, val_loader, device)
        val_mae = val_m.get('aqi_mae', 1e9)

        print(f"Epoch {epoch+1}/{cfg['epochs']} | loss={epoch_loss/n:.4f} | val_mae={val_mae:.4f} | time={t1-t0:.1f}s")

        if val_mae < best_val:
            best_val = val_mae
            torch.save(model.state_dict(), Path(cfg['save_dir'])/"best_litefusion.pth")

    # -------------------
    # Test evaluation
    # -------------------
    test_csv = cfg.get('test_csv')
    if test_csv and Path(test_csv).exists():
        test_loader = DataLoader(
            AQIMultiDataset(test_csv, img_size=cfg['img_size']),
            batch_size=cfg['batch_size'],
            shuffle=False,
            num_workers=cfg['num_workers'],
            collate_fn=collate_batch
        )
        model.load_state_dict(torch.load(Path(cfg['save_dir'])/"best_litefusion.pth", map_location=device))
        test_metrics = evaluate(model, test_loader, device)
        print("Test metrics:", test_metrics)

# ---------------------------
# CLI
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    cfg = DEFAULT_CFG.copy()
    if args.config:
        import yaml
        cfg.update(yaml.safe_load(open(args.config, "r")))

    pr = Path(cfg["project_root"])
    for k in ["train_csv", "val_csv", "test_csv", "master_csv"]:
        if isinstance(cfg[k], str) and not Path(cfg[k]).is_absolute():
            cfg[k] = str(pr / cfg[k])

    print("CONFIG:\n", json.dumps(cfg, indent=2))
    train(cfg)

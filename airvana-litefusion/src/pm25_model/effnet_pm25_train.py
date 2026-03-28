#!/usr/bin/env python3
"""
EfficientNet-B0 — PM2.5 Regression
5-Fold CV + Early Stopping
PM25Vision dataset preprocessed to 128×128 images.

Training data:
  data/pm25vision/metadata_pm25_train.csv
Images:
  data/images/128_pm25/train/

Output:
  models/pm25/effnet_foldX.pt
  models/pm25/effnet_ensemble.json
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from PIL import Image

# ------------------------------------------------------
# Dataset
# ------------------------------------------------------
class PM25Dataset(Dataset):
    def __init__(self, df, img_root):
        self.df = df
        self.img_root = img_root
        self.tr = transforms.Compose([
            transforms.Resize((128,128)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406],
                                 std=[0.229,0.224,0.225]),
        ])

        self.val_tr = transforms.Compose([
            transforms.Resize((128,128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406],
                                 std=[0.229,0.224,0.225]),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row["image_path"]
        y = float(row["pm25"])

        img = Image.open(path).convert("RGB")
        # choose correct transform based on mode
        if self.df.mode == "train":
            img = self.tr(img)
        else:
            img = self.val_tr(img)
        return img, torch.tensor([y], dtype=torch.float32)

# Mark mode attribute later
# ------------------------------------------------------
# Model
# ------------------------------------------------------
class EffNetReg(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_f = self.base.classifier[1].in_features
        self.base.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_f, 1)
        )

    def forward(self, x):
        return self.base(x)

# ------------------------------------------------------
# Train / Eval
# ------------------------------------------------------
def train_one_epoch(model, loader, opt, crit, device):
    model.train()
    total = 0
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        opt.zero_grad()
        pred = model(x)
        loss = crit(pred, y)
        loss.backward()
        opt.step()
        total += loss.item() * x.size(0)
    return total / len(loader.dataset)

def eval_epoch(model, loader, crit, device):
    model.eval()
    total = 0
    mae_t = 0
    with torch.no_grad():
        for x,y in loader:
            x,y = x.to(device), y.to(device)
            pred = model(x)
            loss = crit(pred, y)
            total += loss.item() * x.size(0)
            mae_t += torch.abs(pred - y).sum().item()
    return total / len(loader.dataset), mae_t / len(loader.dataset)

# ------------------------------------------------------
# Main 5-Fold Training
# ------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta", default="data/pm25vision/metadata_pm25_train.csv")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--out", default="models/pm25")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    df = pd.read_csv(args.meta)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    K = 5
    fold_size = len(df) // K

    device = torch.device("mps" if torch.backends.mps.is_available() else
                          "cuda" if torch.cuda.is_available() else "cpu")

    print("Device:", device)

    fold_paths = []

    for fold in range(K):
        print(f"\n===== FOLD {fold+1}/{K} =====")

        val_start = fold * fold_size
        val_end = (fold+1) * fold_size

        df_val = df.iloc[val_start:val_end].copy()
        df_train = pd.concat([df.iloc[:val_start], df.iloc[val_end:]]).copy()

        df_train.mode = "train"
        df_val.mode = "val"

        train_ds = PM25Dataset(df_train, None)
        val_ds   = PM25Dataset(df_val, None)

        train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=4)
        val_loader   = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=4)

        model = EffNetReg().to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        crit = nn.SmoothL1Loss(beta=1.0)

        best = float("inf")
        patience = 6
        patience_c = 0

        out_path = f"{args.out}/effnet_pm25_fold{fold}.pt"

        for ep in range(1, args.epochs+1):
            tr = train_one_epoch(model, train_loader, opt, crit, device)
            vl, mae = eval_epoch(model, val_loader, crit, device)

            print(f"[Fold {fold+1}] Epoch {ep}/{args.epochs} | "
                  f"Train {tr:.3f} | Val {vl:.3f} | MAE {mae:.2f}")

            if vl < best:
                best = vl
                patience_c = 0
                torch.save(model.state_dict(), out_path)
                print("  ✓ Saved best")
            else:
                patience_c += 1
                if patience_c >= patience:
                    print("  → Early stop")
                    break

        fold_paths.append(out_path)

    # Ensemble list
    with open(f"{args.out}/effnet_ensemble.json", "w") as f:
        json.dump({"models": fold_paths}, f, indent=2)

    print("\nTraining complete.")
    print("Best models:", fold_paths)


if __name__ == "__main__":
    main()

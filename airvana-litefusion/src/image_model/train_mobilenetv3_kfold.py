#!/usr/bin/env python3
"""
train_mobilenetv3_kfold.py

Train MobileNetV3-Small for AQI regression using:
- data/metadata_sat_features.csv
- 5-fold cross validation
- CPU-only optimized (MacBook Air safe)

Outputs:
models/mobilenet/mnv3_fold{i}.pt
results/mobilenet/mnv3_fold_metrics.txt
"""

import os
import argparse
import numpy as np
import pandas as pd
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import mobilenet_v3_small

from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from PIL import Image


# -----------------------
# Dataset
# -----------------------

class AQIImageDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.paths = df["image_path"].tolist()

        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        try:
            img = Image.open(path).convert("RGB")
            img = img.resize((128, 128))
            x = torch.tensor(np.array(img)).float() / 255.0
            x = x.permute(2, 0, 1)
            x = (x - self.mean) / self.std
        except:
            x = torch.zeros(3, 128, 128)
        return x


# -----------------------
# Model
# -----------------------

def build_model():
    model = mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, 1)
    return model


# -----------------------
# Training
# -----------------------

def train_one_fold(model, train_idx, val_idx, X, y, device, fold, out_dir):

    train_ds = [(X[i], y[i]) for i in train_idx]
    val_ds = [(X[i], y[i]) for i in val_idx]

    def collate(batch):
        xs, ys = zip(*batch)
        return torch.stack(xs), torch.tensor(ys).float()

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,
                              collate_fn=collate, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=32,
                            collate_fn=collate, num_workers=0)

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.HuberLoss(delta=1.0)

    best_rmse = float("inf")
    best_model_path = os.path.join(out_dir, f"mnv3_fold{fold}.pt")

    for epoch in range(12):  # CPU-safe
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            pred = model(batch_x).squeeze(1)
            loss = loss_fn(pred, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        preds = []
        truths = []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                p = model(batch_x).squeeze(1).cpu().numpy()
                preds.extend(p)
                truths.extend(batch_y.numpy())

        preds = np.array(preds)
        truths = np.array(truths)
        rmse = np.sqrt(((preds - truths) ** 2).mean())

        if rmse < best_rmse:
            best_rmse = rmse
            torch.save(model.state_dict(), best_model_path)

        print(f"Fold {fold} | Epoch {epoch+1}/12 | RMSE={rmse:.3f}")

    return preds, truths


# -----------------------
# Main
# -----------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.metadata)
    df = df[df["aqi_continuous"].notna()].reset_index(drop=True)

    print("Samples:", len(df))

    X = AQIImageDataset(df)
    y = df["aqi_continuous"].values.astype(float)

    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cpu")

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    maes, rmses, r2s = [], [], []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"\n===== Fold {fold} =====")
        model = build_model()
        preds, truths = train_one_fold(
            model, train_idx, val_idx, X, y, device, fold, args.out_dir
        )

        mae = mean_absolute_error(truths, preds)
        rmse = np.sqrt(((preds - truths) ** 2).mean())
        r2 = r2_score(truths, preds)

        maes.append(mae)
        rmses.append(rmse)
        r2s.append(r2)

        print(f"Fold {fold} metrics: MAE={mae:.3f} | RMSE={rmse:.3f} | R2={r2:.3f}")

    # Save CV summary
    report = os.path.join(args.out_dir, "mnv3_fold_metrics.txt")
    with open(report, "w") as f:
        f.write(f"MAE: {np.mean(maes):.3f} ± {np.std(maes):.3f}\n")
        f.write(f"RMSE: {np.mean(rmses):.3f} ± {np.std(rmses):.3f}\n")
        f.write(f"R2: {np.mean(r2s):.3f} ± {np.std(r2s):.3f}\n")

    print("\nDone. Metrics saved:", report)


if __name__ == "__main__":
    main()

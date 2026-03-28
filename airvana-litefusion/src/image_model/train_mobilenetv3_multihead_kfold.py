#!/usr/bin/env python3
"""
train_mobilenetv3_multihead_kfold.py

Multi-task MobileNetV3:
  - Regression: AQI (aqi_continuous)
  - Classification: scene_type (clear / smog / fog / night)

Uses:
  - data/metadata_scene_labels.csv   (from tag_scene_labels.py)

Saves:
  - models/mobilenet/mnv3_multi_fold{0..4}.pt
  - results/mobilenet/mnv3_multi_fold_metrics.txt
"""

import os
import argparse
from typing import Tuple, List

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler

from torchvision import transforms
from torchvision.models import mobilenet_v3_small

from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from PIL import Image


# ---------------------------
# Dataset
# ---------------------------

class AQISceneDataset(Dataset):
    def __init__(self, csv_path: str, root_dir: str = ".", transform=None):
        self.root_dir = root_dir
        self.transform = transform

        df = pd.read_csv(csv_path)

        # keep rows with valid AQI + valid scene label
        mask = df["aqi_continuous"].notna() & (df["scene_label"] >= 0)
        self.df = df[mask].reset_index(drop=True)

        if "image_path" not in self.df.columns:
            raise ValueError("metadata must contain 'image_path' column")

        print(f"Loaded {len(self.df)} samples with scene labels from {csv_path}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = row["image_path"]

        # handle relative paths
        if not os.path.isabs(img_path):
            img_path = os.path.join(self.root_dir, img_path)

        # load image
        img = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        aqi = float(row["aqi_continuous"])
        scene_label = int(row["scene_label"])

        return img, torch.tensor(aqi, dtype=torch.float32), torch.tensor(scene_label, dtype=torch.long)


# ---------------------------
# Model
# ---------------------------

class MobileNetMultiHead(nn.Module):
    """
    MobileNetV3-Small backbone with:
      - Regression head (AQI)
      - Classification head (scene: clear/smog/fog/night)

    Note: This is a NEW architecture (separate from old single-head models).
    """

    def __init__(self, num_scene_classes: int = 4):
        super().__init__()

        self.mnv3 = mobilenet_v3_small(weights=None)  # training from scratch on your dataset
        in_features = self.mnv3.classifier[3].in_features

        # Regression head replaces original classifier[3]
        self.mnv3.classifier[3] = nn.Linear(in_features, 1)

        # Extra classification head for scene
        self.scene_head = nn.Linear(in_features, num_scene_classes)

    def forward(self, x):
        # replicate mobilenet_v3_small forward, but tap into penultimate features
        x = self.mnv3.features(x)
        x = self.mnv3.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.mnv3.classifier[0](x)
        x = self.mnv3.classifier[1](x)
        x = self.mnv3.classifier[2](x)
        feats = x

        aqi = self.mnv3.classifier[3](feats)        # (B,1)
        scene_logits = self.scene_head(feats)       # (B,4)

        return aqi.squeeze(1), scene_logits         # (B,), (B,4)


# ---------------------------
# Metrics helpers
# ---------------------------

def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2


# ---------------------------
# Training routine
# ---------------------------

def train_one_fold(
    fold: int,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    lambda_scene: float
) -> Tuple[float, float, float, float]:
    """
    Train one fold and return:
      - best_val_mae, best_val_rmse, best_val_r2, best_val_scene_acc
    """

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    huber = nn.SmoothL1Loss()
    ce = nn.CrossEntropyLoss()

    best_rmse = float("inf")
    best_metrics = (float("inf"), float("inf"), -1.0, 0.0)  # mae, rmse, r2, scene_acc
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for imgs, aqi, scene_label in train_loader:
            imgs = imgs.to(device)
            aqi = aqi.to(device)
            scene_label = scene_label.to(device)

            optimizer.zero_grad()
            pred_aqi, scene_logits = model(imgs)

            loss_reg = huber(pred_aqi, aqi)

            # classification loss
            loss_scene = ce(scene_logits, scene_label)

            loss = loss_reg + lambda_scene * loss_scene
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)

        avg_train_loss = running_loss / len(train_loader.dataset)

        # ---- validation ----
        model.eval()
        all_y_true, all_y_pred = [], []
        correct_scene, total_scene = 0, 0

        with torch.no_grad():
            for imgs, aqi, scene_label in val_loader:
                imgs = imgs.to(device)
                aqi = aqi.to(device)
                scene_label = scene_label.to(device)

                pred_aqi, scene_logits = model(imgs)

                all_y_true.append(aqi.cpu().numpy())
                all_y_pred.append(pred_aqi.cpu().numpy())

                preds_scene = torch.argmax(scene_logits, dim=1)
                correct_scene += (preds_scene == scene_label).sum().item()
                total_scene += scene_label.size(0)

        y_true = np.concatenate(all_y_true)
        y_pred = np.concatenate(all_y_pred)
        mae, rmse, r2 = compute_regression_metrics(y_true, y_pred)
        scene_acc = correct_scene / max(total_scene, 1)

        print(
            f"[Fold {fold}] Epoch {epoch}/{epochs} "
            f"TrainLoss={avg_train_loss:.4f} "
            f"Val MAE={mae:.3f} RMSE={rmse:.3f} R2={r2:.3f} SceneAcc={scene_acc:.3f}"
        )

        # track best by RMSE
        if rmse < best_rmse:
            best_rmse = rmse
            best_metrics = (mae, rmse, r2, scene_acc)
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    return best_metrics, best_state


# ---------------------------
# Main
# ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--metadata",
        default="data/metadata_scene_labels.csv",
        help="Path to scene-labeled metadata CSV"
    )
    ap.add_argument(
        "--epochs",
        type=int,
        default=12,
        help="Number of epochs per fold (default: 12)"
    )
    ap.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size (default: 32)"
    )
    ap.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate (default: 1e-4)"
    )
    ap.add_argument(
        "--lambda_scene",
        type=float,
        default=0.2,
        help="Weight for scene classification loss"
    )
    ap.add_argument(
        "--num_folds",
        type=int,
        default=5,
        help="Number of folds (default: 5)"
    )
    args = ap.parse_args()

    # device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # transforms (match predictor: resize 128 + ImageNet norm)
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    dataset = AQISceneDataset(
        csv_path=args.metadata,
        root_dir=".",
        transform=transform,
    )

    n_samples = len(dataset)
    indices = np.arange(n_samples)

    kf = KFold(
        n_splits=args.num_folds,
        shuffle=True,
        random_state=42
    )

    os.makedirs("models/mobilenet", exist_ok=True)
    os.makedirs("results/mobilenet", exist_ok=True)

    fold_metrics: List[Tuple[float, float, float, float]] = []

    print(f"\n==== Training {args.num_folds}-Fold MobileNetV3 Multi-Head ====\n")

    for fold, (train_idx, val_idx) in enumerate(kf.split(indices), start=0):
        print(f"\n--- Fold {fold} ---")

        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)

        train_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=train_sampler,
            num_workers=0,
            pin_memory=False,
        )
        val_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=val_sampler,
            num_workers=0,
            pin_memory=False,
        )

        model = MobileNetMultiHead(num_scene_classes=4)

        best_metrics, best_state = train_one_fold(
            fold=fold,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            lambda_scene=args.lambda_scene,
        )

        mae, rmse, r2, scene_acc = best_metrics
        fold_metrics.append(best_metrics)
        print(
            f"[Fold {fold}] BEST → MAE={mae:.3f} RMSE={rmse:.3f} "
            f"R2={r2:.3f} SceneAcc={scene_acc:.3f}"
        )

        # save best model for this fold
        out_path = f"models/mobilenet/mnv3_multi_fold{fold}.pt"
        torch.save(best_state, out_path)
        print("Saved best fold model →", out_path)

    # summary
    maes = [m[0] for m in fold_metrics]
    rmses = [m[1] for m in fold_metrics]
    r2s = [m[2] for m in fold_metrics]
    accs = [m[3] for m in fold_metrics]

    mae_mean, mae_std = np.mean(maes), np.std(maes)
    rmse_mean, rmse_std = np.mean(rmses), np.std(rmses)
    r2_mean, r2_std = np.mean(r2s), np.std(r2s)
    acc_mean, acc_std = np.mean(accs), np.std(accs)

    print("\n==== CV Summary (MobileNet Multi-Head) ====")
    print(f"MAE  : {mae_mean:.3f} ± {mae_std:.3f}")
    print(f"RMSE : {rmse_mean:.3f} ± {rmse_std:.3f}")
    print(f"R²   : {r2_mean:.3f} ± {r2_std:.3f}")
    print(f"SceneAcc: {acc_mean:.3f} ± {acc_std:.3f}")

    metrics_path = "results/mobilenet/mnv3_multi_fold_metrics.txt"
    with open(metrics_path, "w") as f:
        f.write(f"MAE_mean: {mae_mean:.4f}\n")
        f.write(f"MAE_std: {mae_std:.4f}\n")
        f.write(f"RMSE_mean: {rmse_mean:.4f}\n")
        f.write(f"RMSE_std: {rmse_std:.4f}\n")
        f.write(f"R2_mean: {r2_mean:.4f}\n")
        f.write(f"R2_std: {r2_std:.4f}\n")
        f.write(f"SceneAcc_mean: {acc_mean:.4f}\n")
        f.write(f"SceneAcc_std: {acc_std:.4f}\n")

    print("\nCV metrics saved:", metrics_path)
    print("\n==== DONE MULTI-HEAD TRAINING ====")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import os
import json
import argparse
import random

import numpy as np
import pandas as pd

from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import mobilenet_v3_small

from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error


# -------------------------------------------------------------
# Dataset
# -------------------------------------------------------------
class TraqidNightDataset(Dataset):
    def __init__(self, df, img_root=".", num_cols=None, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_root = img_root
        self.num_cols = num_cols or []
        self.transform = transform or transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_path = row["image_path"]
        if not os.path.isabs(img_path):
            img_path = os.path.join(self.img_root, img_path)

        img = Image.open(img_path).convert("RGB")
        x_img = self.transform(img)

        num_feats = torch.tensor(
            row[self.num_cols].values.astype("float32"),
            dtype=torch.float32,
        )

        y = torch.tensor([float(row["aqi"])], dtype=torch.float32)

        return x_img, num_feats, y


# -------------------------------------------------------------
# Model: MobileNetV3 + numeric fusion, AQI regression
# -------------------------------------------------------------
class MobileNetV3TraqidAQI(nn.Module):
    def __init__(self, num_features: int):
        super().__init__()
        base = mobilenet_v3_small(weights="DEFAULT")

        self.features = base.features
        self.pool = nn.AdaptiveAvgPool2d(1)

        # 576 is the final channel size for mobilenet_v3_small
        self.img_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(576, 256),
            nn.ReLU(),
        )

        self.num_head = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.ReLU(),
        )

        self.regressor = nn.Sequential(
            nn.Linear(256 + 64, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, img, num_feats):
        x = self.features(img)
        x = self.pool(x)
        x = self.img_head(x)

        n = self.num_head(num_feats)

        h = torch.cat([x, n], dim=1)
        out = self.regressor(h)
        return out


# -------------------------------------------------------------
# Helpers
# -------------------------------------------------------------
def find_image_file(image_id, root_dir):
    image_id = str(image_id)

    exts = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]
    subdirs = ["Front", "Rear", ""]

    # Direct name match: <id>.ext
    for sub in subdirs:
        d = os.path.join(root_dir, sub) if sub else root_dir
        if not os.path.isdir(d):
            continue
        for ext in exts:
            cand = os.path.join(d, image_id + ext)
            if os.path.exists(cand):
                return os.path.relpath(cand, ".")

    # Fallback: first file starting with id
    for sub in subdirs:
        d = os.path.join(root_dir, sub) if sub else root_dir
        if not os.path.isdir(d):
            continue
        for fname in os.listdir(d):
            if fname.startswith(image_id):
                return os.path.relpath(os.path.join(d, fname), ".")

    return None


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    n_samples = 0

    for imgs, nums, y in loader:
        imgs = imgs.to(device)
        nums = nums.to(device)
        y = y.to(device).view(-1)

        optimizer.zero_grad()
        preds = model(imgs, nums).view(-1)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()

        bs = imgs.size(0)
        total_loss += loss.item() * bs
        n_samples += bs

    return total_loss / max(1, n_samples)


def eval_epoch(model, loader, device):
    model.eval()
    all_y = []
    all_pred = []

    with torch.no_grad():
        for imgs, nums, y in loader:
            imgs = imgs.to(device)
            nums = nums.to(device)
            y = y.to(device).view(-1)

            preds = model(imgs, nums).view(-1)

            all_y.extend(y.cpu().numpy().tolist())
            all_pred.extend(preds.cpu().numpy().tolist())

    all_y = np.array(all_y, dtype=np.float32)
    all_pred = np.array(all_pred, dtype=np.float32)

    mae = mean_absolute_error(all_y, all_pred)
    rmse = np.sqrt(mean_squared_error(all_y, all_pred))

    return mae, rmse


# -------------------------------------------------------------
# Main
# -------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata",
                    default="data/TRAQID_sample/TRAQID.csv")
    ap.add_argument("--img_root",
                    default="data/TRAQID_sample/Images/2")
    ap.add_argument("--out_dir",
                    default="models/traqid_night_aqi")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Seeds
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available()
                          else "cpu")
    print("Using device:", device)

    # ---------------------------------------------------------
    # Load TRAQID metadata, keep Sequence == 2 (night route)
    # ---------------------------------------------------------
    df = pd.read_csv(args.metadata)

    # If column names differ, adjust here
    needed_cols = [
        "Image",
        "Sequence",
        "Temperature",
        "Humidity",
        "Season",
        "Day_or_Night",
        "PM2.5",
        "PM10",
        "aqi",
    ]
    for c in needed_cols:
        if c not in df.columns:
            raise ValueError(f"Required column '{c}' missing in {args.metadata}")

    df = df[df["Sequence"] == 2].copy()
    df = df.dropna(subset=["aqi"]).reset_index(drop=True)

    # Map image id -> file path
    paths = []
    for _, row in df.iterrows():
        img_id = row["Image"]
        p = find_image_file(img_id, args.img_root)
        paths.append(p)

    df["image_path"] = paths
    df = df.dropna(subset=["image_path"]).reset_index(drop=True)

    if len(df) == 0:
        raise RuntimeError("No images found for TRAQID Sequence==2.")

    print(f"Total TRAQID night samples with images: {len(df)}")

    # ---------------------------------------------------------
    # Build numeric feature columns
    # ---------------------------------------------------------
    season_map = {name: i for i, name in enumerate(sorted(df["Season"].dropna().unique()))}
    df["season_idx"] = df["Season"].map(season_map).astype("int64")

    day_map = {"Day": 0, "Night": 1}
    df["day_idx"] = df["Day_or_Night"].map(day_map).fillna(0).astype("int64")

    base_num_cols = ["PM2.5", "PM10", "Temperature", "Humidity",
                     "season_idx", "day_idx"]

    # Global normalization stats (saved for inference later)
    norm_stats = {}
    for c in base_num_cols:
        mean = float(df[c].mean())
        std = float(df[c].std() if df[c].std() > 1e-6 else 1.0)
        df[c + "_norm"] = (df[c] - mean) / std
        norm_stats[c] = {"mean": mean, "std": std}

    norm_stats_path = os.path.join(args.out_dir, "traqid_night_norm_stats.json")
    with open(norm_stats_path, "w") as f:
        json.dump(norm_stats, f, indent=2)
    print(f"Saved normalization stats → {norm_stats_path}")

    num_cols = [c + "_norm" for c in base_num_cols]

    # ---------------------------------------------------------
    # K-Fold setup
    # ---------------------------------------------------------
    n = len(df)
    indices = np.arange(n)

    kf = KFold(n_splits=args.folds, shuffle=True, random_state=42)

    all_test_mae = []
    all_test_rmse = []

    transform_train = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.15, contrast=0.15,
                               saturation=0.15),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    transform_val = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    for fold, (train_idx, test_idx) in enumerate(kf.split(indices), start=0):
        print(f"\n==== Fold {fold} ====")

        # Approximate 75/15/10 → use part of train set as validation
        rng = np.random.RandomState(42 + fold)
        shuffled_train = rng.permutation(train_idx)

        val_size = int(0.15 * n)
        val_size = min(val_size, len(shuffled_train) // 3)  # safety

        val_idx = shuffled_train[:val_size]
        train_idx_final = shuffled_train[val_size:]

        df_train = df.iloc[train_idx_final].reset_index(drop=True)
        df_val = df.iloc[val_idx].reset_index(drop=True)
        df_test = df.iloc[test_idx].reset_index(drop=True)

        print(f"Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}")

        ds_train = TraqidNightDataset(df_train, img_root=".", num_cols=num_cols,
                                      transform=transform_train)
        ds_val = TraqidNightDataset(df_val, img_root=".", num_cols=num_cols,
                                    transform=transform_val)
        ds_test = TraqidNightDataset(df_test, img_root=".", num_cols=num_cols,
                                     transform=transform_val)

        loader_train = DataLoader(
            ds_train, batch_size=args.batch_size, shuffle=True,
            num_workers=2, pin_memory=True
        )
        loader_val = DataLoader(
            ds_val, batch_size=args.batch_size, shuffle=False,
            num_workers=2, pin_memory=True
        )
        loader_test = DataLoader(
            ds_test, batch_size=args.batch_size, shuffle=False,
            num_workers=2, pin_memory=True
        )

        model = MobileNetV3TraqidAQI(num_features=len(num_cols)).to(device)

        criterion = nn.HuberLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        best_val_rmse = float("inf")
        best_path = os.path.join(
            args.out_dir, f"mnv3_traqid_night_fold{fold}.pt"
        )

        for epoch in range(1, args.epochs + 1):
            train_loss = train_one_epoch(
                model, loader_train, optimizer, criterion, device
            )
            val_mae, val_rmse = eval_epoch(model, loader_val, device)

            print(
                f"Epoch {epoch}/{args.epochs} | "
                f"train_loss={train_loss:.4f} | "
                f"val_rmse={val_rmse:.3f} | val_mae={val_mae:.3f}"
            )

            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                torch.save(model.state_dict(), best_path)
                print(f"  ✓ Saved best model → {best_path}")

        # Load best weights and evaluate on test set
        best_model = MobileNetV3TraqidAQI(num_features=len(num_cols)).to(device)
        best_model.load_state_dict(torch.load(best_path, map_location=device))

        test_mae, test_rmse = eval_epoch(best_model, loader_test, device)
        print(
            f"[Fold {fold}] Test RMSE={test_rmse:.3f} | "
            f"Test MAE={test_mae:.3f}"
        )

        all_test_mae.append(test_mae)
        all_test_rmse.append(test_rmse)

    print("\n==== TRAQID Night AQI Regression CV Summary ====")
    print(
        f"Test RMSE: {np.mean(all_test_rmse):.3f} ± {np.std(all_test_rmse):.3f}"
    )
    print(
        f"Test MAE : {np.mean(all_test_mae):.3f} ± {np.std(all_test_mae):.3f}"
    )


if __name__ == "__main__":
    main()

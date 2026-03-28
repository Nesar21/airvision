import os
import argparse
import random
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.models import mobilenet_v3_small

# -------------------------------------------------------------
# Dataset
# -------------------------------------------------------------
class AQIDataset(Dataset):
    def __init__(self, df, img_root):
        self.df = df
        self.img_root = img_root

        self.transform = T.Compose([
            T.Resize((128, 128)),
            T.RandomHorizontalFlip(),
            T.RandomRotation(5),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_path = row["image_path"]
        aqi = torch.tensor([float(row["aqi_continuous"])], dtype=torch.float32)

        full_path = img_path
        if not os.path.exists(full_path):
            # fallback: manual join if metadata path is relative
            full_path = os.path.join(self.img_root, os.path.basename(img_path))

        img = Image.open(full_path).convert("RGB")
        img = self.transform(img)

        return img, aqi


# -------------------------------------------------------------
# MobileNetV3 Regressor — FIXED VERSION
# -------------------------------------------------------------
class MobileNetV3Regressor(nn.Module):
    def __init__(self):
        super().__init__()
        base = mobilenet_v3_small(weights="DEFAULT")

        # keep feature extractor ONLY
        self.features = base.features
        self.pool = nn.AdaptiveAvgPool2d(1)

        # MobileNetV3-small last output = 576 dims
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(576, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.regressor(x)
        return x


# -------------------------------------------------------------
# Train epoch
# -------------------------------------------------------------
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0

    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        preds = model(imgs)

        loss = criterion(preds, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


# -------------------------------------------------------------
# Validation epoch
# -------------------------------------------------------------
def val_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            preds = model(imgs)
            loss = criterion(preds, labels)

            total_loss += loss.item()

    return total_loss / len(loader)


# -------------------------------------------------------------
# Main
# -------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--img_root", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--out", required=True)

    args = parser.parse_args()

    # Load metadata
    df = pd.read_csv(args.metadata)
    df = df.dropna(subset=["aqi_continuous", "image_path"]).reset_index(drop=True)

    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Train/val split (90/10)
    split_idx = int(0.9 * len(df))
    train_df = df[:split_idx]
    val_df = df[split_idx:]

    print(f"Total images: {len(df)}")
    print(f"Train: {len(train_df)},  Val: {len(val_df)}")

    # Datasets
    train_ds = AQIDataset(train_df, args.img_root)
    val_ds = AQIDataset(val_df, args.img_root)

    # Loaders
    train_loader = DataLoader(train_ds, batch_size=args.batch,
                              shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch,
                            shuffle=False, num_workers=2, pin_memory=True)

    # Model
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print("Using device:", device)

    model = MobileNetV3Regressor().to(device)

    # Loss + Optimizer
    criterion = nn.HuberLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    best_loss = float("inf")

    # Training loop
    for epoch in range(1, args.epochs + 1):
        tr_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        vl_loss = val_epoch(model, val_loader, criterion, device)

        print(f"[Epoch {epoch}/{args.epochs}] Train: {tr_loss:.4f} | Val: {vl_loss:.4f}")

        # Save best model
        if vl_loss < best_loss:
            best_loss = vl_loss
            torch.save(model.state_dict(), args.out)
            print("  ✓ Saved best model")

    print("Training complete.")


if __name__ == "__main__":
    main()

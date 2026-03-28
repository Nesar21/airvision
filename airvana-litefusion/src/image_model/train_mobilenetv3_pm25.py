#!/usr/bin/env python3
import os
import argparse
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

# -----------------------------------------------------
# Dataset
# -----------------------------------------------------
class PM25Dataset(Dataset):
    def __init__(self, df, img_root, augment=False):
        self.df = df
        self.img_root = img_root

        if augment:
            self.transforms = transforms.Compose([
                transforms.Resize((128,128)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.ToTensor(),
            ])
        else:
            self.transforms = transforms.Compose([
                transforms.Resize((128,128)),
                transforms.ToTensor(),
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["image_path"]

        img = Image.open(img_path).convert("RGB")
        img = self.transforms(img)

        pm25 = torch.tensor([row["pm25"]], dtype=torch.float32)

        return img, pm25


# -----------------------------------------------------
# Train / Val loops
# -----------------------------------------------------
def train_epoch(model, loader, criterion, opt, device):
    model.train()
    total = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)

        opt.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        opt.step()

        total += loss.item() * x.size(0)
    return total / len(loader.dataset)


def val_epoch(model, loader, criterion, device):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = criterion(pred, y)
            total += loss.item() * x.size(0)
    return total / len(loader.dataset)


# -----------------------------------------------------
# Main
# -----------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_meta", required=True)
    parser.add_argument("--test_meta", required=True)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    # Load metadata
    train_df = pd.read_csv(args.train_meta)
    test_df  = pd.read_csv(args.test_meta)

    # Dataset roots already embedded via absolute paths in metadata
    train_ds = PM25Dataset(train_df, "", augment=True)
    test_ds  = PM25Dataset(test_df, "", augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_loader   = DataLoader(test_ds,  batch_size=args.batch)

    # Device
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print("Using:", device)

    # Model
    base = models.mobilenet_v3_small(weights="DEFAULT")

    # Replace classifier → 1 regression output
    in_features = base.classifier[3].in_features
    base.classifier[3] = nn.Linear(in_features, 1)

    model = base.to(device)

    # Loss / Optimizer
    criterion = nn.HuberLoss()
    opt = optim.Adam(model.parameters(), lr=1e-4)

    best = float("inf")

    # Training loop
    for epoch in range(1, args.epochs + 1):
        tr = train_epoch(model, train_loader, criterion, opt, device)
        va = val_epoch(model, val_loader, criterion, device)

        print(f"[Epoch {epoch}/{args.epochs}] Train: {tr:.4f} | Val: {va:.4f}")

        if va < best:
            best = va
            torch.save(model.state_dict(), args.out)
            print("  ✓ Saved best model")

    print("Training complete.")


if __name__ == "__main__":
    main()

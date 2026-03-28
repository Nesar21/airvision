#!/usr/bin/env python3
import os
import argparse

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import mobilenet_v3_small
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score


class HazeDataset(Dataset):
    def __init__(self, df, img_base=".", transform=None):
        self.df = df.reset_index(drop=True)
        self.img_base = img_base
        self.transform = transform or transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        from PIL import Image

        row = self.df.iloc[idx]
        path = row["image_path"]
        if not os.path.isabs(path):
            img_path = os.path.join(self.img_base, path)
        else:
            img_path = path

        img = Image.open(img_path).convert("RGB")
        x = self.transform(img)
        y = int(row["haze_label"])
        return x, y


def build_model(device):
    # Start from your regression backbone weights
    model = mobilenet_v3_small(weights=None)
    # Load regression backbone if available
    reg_path = "models/mobilenet/mnv3_regression.pt"
    if os.path.exists(reg_path):
        state = torch.load(reg_path, map_location=device)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print("Loaded regression backbone, missing keys:", missing, "unexpected:", unexpected)
    else:
        print("WARNING: mnv3_regression.pt not found, training haze model from scratch.")

    # Replace final classifier layer with 2-class head
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, 2)
    model.to(device)
    return model


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x.size(0)

    return running_loss / len(loader.dataset)


def eval_model(model, loader, device):
    model.eval()
    all_y = []
    all_pred = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            preds = torch.argmax(logits, dim=1)
            all_y.extend(y.cpu().numpy().tolist())
            all_pred.extend(preds.cpu().numpy().tolist())

    acc = accuracy_score(all_y, all_pred)
    f1 = f1_score(all_y, all_pred)
    return acc, f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", default="data/metadata_scene_haze.csv")
    ap.add_argument("--img_base", default=".")
    ap.add_argument("--out_dir", default="models/mobilenet_haze")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("Using device:", device)

    df = pd.read_csv(args.metadata)

    # Use only labeled rows (0 / 1)
    df = df[df["haze_label"].isin([0, 1])].reset_index(drop=True)
    print("Total labeled samples:", len(df))
    print(df["haze_label"].value_counts())

    X = df["image_path"].values
    y = df["haze_label"].values

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)

    transforms_train = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
    ])

    transforms_val = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    all_acc = []
    all_f1 = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=0):
        print(f"\n==== Fold {fold} ====")
        df_tr = df.iloc[tr_idx].reset_index(drop=True)
        df_va = df.iloc[va_idx].reset_index(drop=True)

        ds_tr = HazeDataset(df_tr, img_base=args.img_base, transform=transforms_train)
        ds_va = HazeDataset(df_va, img_base=args.img_base, transform=transforms_val)

        loader_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, num_workers=2)
        loader_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False, num_workers=2)

        model = build_model(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        best_f1 = 0.0
        best_path = os.path.join(args.out_dir, f"mnv3_haze_fold{fold}.pt")

        for epoch in range(args.epochs):
            loss_tr = train_one_epoch(model, loader_tr, optimizer, criterion, device)
            acc_va, f1_va = eval_model(model, loader_va, device)

            print(f"Epoch {epoch+1}/{args.epochs} | "
                  f"loss={loss_tr:.4f} | val_acc={acc_va:.4f} | val_f1={f1_va:.4f}")

            if f1_va > best_f1:
                best_f1 = f1_va
                torch.save(model.state_dict(), best_path)

        print(f"Best F1 for fold {fold}: {best_f1:.4f} (saved → {best_path})")
        all_acc.append(acc_va)
        all_f1.append(best_f1)

    print("\n==== Haze classifier CV summary ====")
    print(f"Acc (last epoch mean): {np.mean(all_acc):.4f} ± {np.std(all_acc):.4f}")
    print(f"F1  (best per fold):   {np.mean(all_f1):.4f} ± {np.std(all_f1):.4f}")


if __name__ == "__main__":
    main()

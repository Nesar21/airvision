#!/usr/bin/env python3
import os
import argparse
import numpy as np
import pandas as pd

from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score

from src.mobilenet.mobilenetv3_haze import MobileNetV3Haze


class HazeMultiSourceDataset(Dataset):
    def __init__(self, df, img_base=".", train=True):
        self.df = df.reset_index(drop=True)
        self.img_base = img_base
        if train:
            self.transform = transforms.Compose([
                transforms.Resize((128, 128)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row["image_path"]
        if not os.path.isabs(path):
            path = os.path.join(self.img_base, path)

        img = Image.open(path).convert("RGB")
        x = self.transform(img)
        y = int(row["haze_label"])
        return x, y


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)


def eval_model(model, loader, device):
    model.eval()
    ys = []
    ps = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            pred = torch.argmax(logits, dim=1)
            ys.extend(y.cpu().numpy())
            ps.extend(pred.cpu().numpy())
    acc = accuracy_score(ys, ps)
    f1 = f1_score(ys, ps)
    return acc, f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", default="data/metadata_haze_multisource.csv")
    ap.add_argument("--img_base", default=".")
    ap.add_argument("--out_dir", default="models/mobilenet_haze_multisource")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--backbone_ckpt", default="models/mobilenet/mnv3_regression.pt")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("Using device:", device)

    df = pd.read_csv(args.metadata)
    df = df[df["haze_label"].isin([0, 1])].reset_index(drop=True)
    print("Total samples:", len(df))
    print(df["haze_label"].value_counts())

    X = df["image_path"].values
    y = df["haze_label"].values

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)

    all_last_acc = []
    all_best_f1 = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=0):
        print(f"\n==== Fold {fold} ====")
        df_tr = df.iloc[tr_idx].reset_index(drop=True)
        df_va = df.iloc[va_idx].reset_index(drop=True)

        ds_tr = HazeMultiSourceDataset(df_tr, img_base=args.img_base, train=True)
        ds_va = HazeMultiSourceDataset(df_va, img_base=args.img_base, train=False)

        dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, num_workers=2)
        dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False, num_workers=2)

        model = MobileNetV3Haze(
            num_classes=2,
            backbone_ckpt=args.backbone_ckpt,
            device=device,
        ).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        best_f1 = 0.0
        last_acc = 0.0
        best_path = os.path.join(args.out_dir, f"mnv3_haze_multisource_fold{fold}.pt")

        for epoch in range(1, args.epochs + 1):
            loss_tr = train_one_epoch(model, dl_tr, optimizer, criterion, device)
            acc_va, f1_va = eval_model(model, dl_va, device)
            last_acc = acc_va

            print(f"Epoch {epoch}/{args.epochs} | "
                  f"loss={loss_tr:.4f} | val_acc={acc_va:.4f} | val_f1={f1_va:.4f}")

            if f1_va > best_f1:
                best_f1 = f1_va
                torch.save(model.state_dict(), best_path)
                print("  ✓ Saved best model →", best_path)

        print(f"Best F1 for fold {fold}: {best_f1:.4f}")
        all_last_acc.append(last_acc)
        all_best_f1.append(best_f1)

    print("\n==== Multi-source haze classifier CV summary ====")
    print(f"Acc (last-epoch mean): {np.mean(all_last_acc):.4f} ± {np.std(all_last_acc):.4f}")
    print(f"F1  (best per fold):   {np.mean(all_best_f1):.4f} ± {np.std(all_best_f1):.4f}")


if __name__ == "__main__":
    main()

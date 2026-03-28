#!/usr/bin/env python3
import os
import json
import argparse
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import mobilenet_v3_small

# ---------------------------------------------------------
# Dataset
# ---------------------------------------------------------
class TraqidImageDataset(Dataset):
    def __init__(self, df, img_root=".", transform=None):
        self.df = df.reset_index(drop=True)
        self.img_root = img_root
        self.transform = transform or transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row["image_path"]
        if not os.path.isabs(path):
            path = os.path.join(self.img_root, path)

        img = Image.open(path).convert("RGB")
        x_img = self.transform(img)
        return x_img, row["Image"]


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
def find_image_file(image_id, root_dir):
    exts = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]
    subdirs = ["Front", "Rear", ""]

    image_id = str(image_id)

    for sub in subdirs:
        d = os.path.join(root_dir, sub) if sub else root_dir
        if not os.path.isdir(d):
            continue
        for ext in exts:
            fp = os.path.join(d, image_id + ext)
            if os.path.exists(fp):
                return os.path.relpath(fp, ".")

    for sub in subdirs:
        d = os.path.join(root_dir, sub) if sub else root_dir
        if not os.path.isdir(d):
            continue
        for fname in os.listdir(d):
            if fname.startswith(image_id):
                return os.path.relpath(os.path.join(d, fname), ".")

    return None


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--img_root", required=True)
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--out", default="data/traqid_embeddings.csv")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    df = pd.read_csv(args.metadata)

    df = df[df["Sequence"] == 2].copy()
    df = df.dropna(subset=["aqi"]).reset_index(drop=True)

    paths = []
    for _, r in df.iterrows():
        p = find_image_file(r["Image"], args.img_root)
        paths.append(p)

    df["image_path"] = paths
    df = df.dropna(subset=["image_path"]).reset_index(drop=True)
    print(f"Images found: {len(df)}")

    season_map = {n: i for i, n in enumerate(sorted(df["Season"].dropna().unique()))}
    df["season_idx"] = df["Season"].map(season_map).astype("int64")

    df["day_idx"] = df["Day_or_Night"].map({"Day": 0, "Night": 1}).fillna(0).astype("int64")

    # Load numeric normalization stats (from regression model)
    norm_path = "models/traqid_night_aqi/traqid_night_norm_stats.json"
    with open(norm_path, "r") as f:
        norm_stats = json.load(f)

    base_num_cols = ["PM2.5", "PM10", "Temperature", "Humidity"]
    for c in base_num_cols:
        df[c + "_norm"] = (
            df[c] - norm_stats[c]["mean"]
        ) / max(1e-6, norm_stats[c]["std"])

    extra_cols = [c + "_norm" for c in base_num_cols] + ["season_idx", "day_idx"]

    # Load MobileNet haze backbone
    model_path = os.path.join(args.model_dir, "mnv3_haze_multisource_fold0.pt")
    state = torch.load(model_path, map_location="cpu")

    base = mobilenet_v3_small(weights="DEFAULT")
    backbone = base.features
    pool = nn.AdaptiveAvgPool2d(1)
    backbone.load_state_dict({k: v for k, v in state.items() if k in backbone.state_dict()}, strict=False)
    backbone.eval()

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    ds = TraqidImageDataset(df, img_root=".", transform=transform)
    loader = DataLoader(ds, batch_size=1, shuffle=False)

    embeddings = []
    device = torch.device("cpu")

    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.to(device)
            x = backbone(imgs)
            x = pool(x).view(1, -1)
            embeddings.append(x.cpu().numpy())

    embeddings = np.concatenate(embeddings, axis=0)
    dim = embeddings.shape[1]
    print(f"Embedding shape: {embeddings.shape}")

    out_df = df[["Image", "image_path", "aqi"] + extra_cols].copy()
    for i in range(dim):
        out_df[f"emb_{i}"] = embeddings[:, i]

    out_df.to_csv(args.out, index=False)

    print(f"Saved embeddings → {args.out}")
    print("Total columns:", len(out_df.columns))


if __name__ == "__main__":
    main()

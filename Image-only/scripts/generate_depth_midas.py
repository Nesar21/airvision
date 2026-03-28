#!/usr/bin/env python3
"""
Stable MiDaS depth generator for macOS (CPU-safe)
Corrects the transform bug causing 5D tensors.
"""

import argparse
import os
import cv2
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

def load_midas(device):
    model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
    model.to(device)
    model.eval()

    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
    transform = midas_transforms.small_transform

    return model, transform

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def normalize_depth(arr):
    arr = arr.astype("float32")
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-6:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)

def main(args):
    device = torch.device("cpu")   # macOS safe
    print("Device:", device)

    df = pd.read_csv(args.csv)
    print("Loaded rows:", len(df))

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    model, transform = load_midas(device)

    rows = df["image_path"].tolist()
    todo = []

    for p in rows:
        bn = Path(p).name
        out_file = out_dir / f"{bn}.npy"
        if not out_file.exists():
            todo.append((p, out_file))

    print("Images needing depth:", len(todo))

    for img_path, out_file in tqdm(todo, ncols=100, desc="Depth"):
        img = cv2.imread(img_path)
        if img is None:
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # ===== FIX: Ensure transform returns 3D tensor =====
        inp = transform(img)
        if inp.ndim == 4:       # shape [1,3,H,W]
            inp = inp.squeeze(0)
        # Now final input is [3,H,W]

        inp = inp.to(device).unsqueeze(0)  # → [1,3,H,W]

        with torch.no_grad():
            pred = model(inp).squeeze().cpu().numpy()

        pred = normalize_depth(pred)
        pred = cv2.resize(pred, (128, 128), interpolation=cv2.INTER_CUBIC)

        np.save(str(out_file), pred)

    print("Done. Output →", out_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()
    main(args)

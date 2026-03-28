#!/usr/bin/env python3
import os
import argparse

import numpy as np
import pandas as pd
from PIL import Image
import cv2


def build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--metadata",
        default="data/metadata_continuous.csv",
        help="Metadata CSV with image_path and aqi_continuous",
    )
    ap.add_argument(
        "--img_base",
        default=".",
        help="Base directory for image_path if paths are relative",
    )
    ap.add_argument(
        "--out",
        default="data/metadata_haze_features.csv",
        help="Output CSV with added image brightness/blur features",
    )
    return ap


def compute_features(img_path):
    try:
        img = Image.open(img_path).convert("RGB")
    except Exception:
        return np.nan, np.nan, np.nan

    img = img.resize((128, 128))
    arr = np.array(img).astype(np.float32) / 255.0

    # Brightness: mean over all pixels
    brightness = float(arr.mean())

    # Convert to gray and compute Laplacian variance as blur measure
    gray = cv2.cvtColor((arr * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lap_var = float(lap.var())

    # Contrast: std dev of gray
    contrast = float(gray.std())

    return brightness, lap_var, contrast


def main():
    args = build_argparser().parse_args()

    df = pd.read_csv(args.metadata)
    if "image_path" not in df.columns:
        raise ValueError("metadata must contain 'image_path' column")

    img_base = args.img_base

    brightness_list = []
    lap_var_list = []
    contrast_list = []

    for idx, row in df.iterrows():
        path = row["image_path"]
        if not isinstance(path, str):
            brightness_list.append(np.nan)
            lap_var_list.append(np.nan)
            contrast_list.append(np.nan)
            continue

        if os.path.isabs(path):
            img_path = path
        else:
            img_path = os.path.join(img_base, path)

        if not os.path.exists(img_path):
            brightness_list.append(np.nan)
            lap_var_list.append(np.nan)
            contrast_list.append(np.nan)
            continue

        b, lv, c = compute_features(img_path)
        brightness_list.append(b)
        lap_var_list.append(lv)
        contrast_list.append(c)

        if (idx + 1) % 1000 == 0:
            print(f"Processed {idx + 1} images...")

    df["img_brightness"] = brightness_list
    df["img_lap_var"] = lap_var_list
    df["img_contrast"] = contrast_list

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print("Saved image haze features →", args.out)


if __name__ == "__main__":
    main()

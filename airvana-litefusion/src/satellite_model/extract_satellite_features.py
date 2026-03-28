#!/usr/bin/env python3
"""
extract_satellite_features.py

Usage (example):
  source venv/bin/activate
  pip install -r scripts/requirements.txt
  python3 src/satellite_model/extract_satellite_features.py \
    --sat_root data/satellite \
    --metadata data/metadata_enriched.csv \
    --out data/metadata_sat_features.csv \
    --sat_col satellite_path \
    --img_base data

Outputs:
 - CSV with extra columns: sat_brightness, sat_blur, sat_color_skew
 - Merges into metadata when satellite_path available and file exists.

Dependencies:
  pillow, numpy, pandas, opencv-python, tqdm
"""

import os
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
from PIL import Image
import pandas as pd
import cv2

# -------------------------
# Small image-stat helpers
# -------------------------
def load_image_rgb(path):
    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            return np.array(im)
    except Exception:
        return None

def brightness_from_rgb(arr):
    # arr: HxWx3 uint8
    # convert to luminance (Y) using Rec. 601 luma coefficients
    if arr is None: 
        return np.nan
    r = arr[..., 0].astype(np.float32)
    g = arr[..., 1].astype(np.float32)
    b = arr[..., 2].astype(np.float32)
    y = 0.299 * r + 0.587 * g + 0.114 * b
    return float(np.mean(y))

def blur_from_rgb(arr):
    # variance of Laplacian on grayscale -> higher = sharper, lower = blurrier
    if arr is None:
        return np.nan
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    var = float(lap.var())
    return var

def color_skew_from_rgb(arr):
    # compute Pearson skewness (3rd standardized moment) per channel and take mean of absolute values
    # fallback to channel mean differences if variance is zero
    if arr is None:
        return np.nan
    arrf = arr.astype(np.float64)
    vals = []
    for c in range(3):
        ch = arrf[..., c].ravel()
        mean = ch.mean()
        std = ch.std()
        if std < 1e-6:
            # degenerate: use (mean - mid)/255
            vals.append((mean - 127.5) / 255.0)
        else:
            m3 = ((ch - mean) ** 3).mean()
            skew = m3 / (std ** 3 + 1e-12)
            vals.append(skew)
    # compress to single scalar while keeping sign info and scale-consistency
    # use mean of channel skews
    return float(np.mean(vals))

# -------------------------
# Main flow
# -------------------------
def find_satellite_path(sat_root: Path, sat_path_value: str, img_base: Path):
    """
    Resolve satellite_path entry (it may be absolute, relative, or just filename).
    Priority:
     1) If sat_path_value is absolute and exists -> return it
     2) If sat_path_value relative to img_base -> return that
     3) If sat_path_value relative to sat_root -> return that
     4) If sat_path_value empty -> return None
    """
    if not sat_path_value or pd.isna(sat_path_value):
        return None
    p = Path(sat_path_value)
    if p.is_absolute() and p.exists():
        return p
    # try relative to repo / img_base
    cand = img_base / p
    if cand.exists():
        return cand
    cand2 = sat_root / p
    if cand2.exists():
        return cand2
    # try common extensions
    for ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
        if not str(p).lower().endswith(ext):
            c = sat_root / (str(p) + ext)
            if c.exists():
                return c
    return None

def process_row_sat(path: Path):
    arr = load_image_rgb(str(path))
    if arr is None:
        return (np.nan, np.nan, np.nan)
    b = brightness_from_rgb(arr)
    bl = blur_from_rgb(arr)
    cs = color_skew_from_rgb(arr)
    return (b, bl, cs)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sat_root", required=True, help="root folder for satellite tiles (data/satellite)")
    parser.add_argument("--metadata", required=True, help="input metadata CSV")
    parser.add_argument("--out", required=True, help="output metadata CSV with satellite features")
    parser.add_argument("--sat_col", default="satellite_path", help="column name in metadata pointing to sat tile (default: satellite_path)")
    parser.add_argument("--img_base", default=".", help="base path to resolve relative paths from metadata")
    parser.add_argument("--skip_existing", action="store_true", help="if true, will not recompute rows that already have features")
    args = parser.parse_args()

    sat_root = Path(args.sat_root)
    meta_path = Path(args.metadata)
    out_path = Path(args.out)
    img_base = Path(args.img_base)

    assert meta_path.exists(), f"metadata not found: {meta_path}"
    df = pd.read_csv(meta_path)

    # add columns if missing
    for c in ("sat_brightness", "sat_blur", "sat_color_skew"):
        if c not in df.columns:
            df[c] = np.nan

    idxs = df.index.tolist()
    print(f"Rows in metadata: {len(idxs)}")

    # Precompute unique satellite entries to avoid duplicates
    resolve_cache = {}
    work = []
    for idx in idxs:
        sat_val = df.at[idx, args.sat_col] if args.sat_col in df.columns else None
        key = str(sat_val) if pd.notna(sat_val) else ""
        if key in resolve_cache:
            continue
        resolve_cache[key] = None
        work.append((idx, key, sat_val))

    # Process each unique sat path
    print(f"Unique satellite keys to resolve: {len(work)}")
    results_cache = {}
    for _, key, sat_val in tqdm(work, desc="resolving and processing"):
        if key == "" or pd.isna(sat_val):
            results_cache[key] = (np.nan, np.nan, np.nan)
            continue
        resolved = find_satellite_path(sat_root, str(sat_val), img_base)
        if resolved is None:
            # not found
            results_cache[key] = (np.nan, np.nan, np.nan)
            continue
        features = process_row_sat(resolved)
        results_cache[key] = features

    # Map results back into dataframe
    for idx in idxs:
        sat_val = df.at[idx, args.sat_col] if args.sat_col in df.columns else None
        key = str(sat_val) if pd.notna(sat_val) else ""
        b, bl, cs = results_cache.get(key, (np.nan, np.nan, np.nan))
        df.at[idx, "sat_brightness"] = b
        df.at[idx, "sat_blur"] = bl
        df.at[idx, "sat_color_skew"] = cs

    # write out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print("Wrote:", out_path)

if __name__ == "__main__":
    main()

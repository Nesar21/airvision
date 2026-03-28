#!/usr/bin/env python3
"""
STEP 1 — Build master_v1.csv (updated)
- Robust TRAQID filename resolver (searches subfolders)
- Safe numeric parsing & mask-friendly schema
- PM2.5 -> AQI conversion
- SAPID category -> soft AQI
- Depth autodetection
Usage:
    python scripts/build_master_mastercsv.py --project_root /Users/nesar/VS/Image-only
"""
import argparse
import os
import sys
import math
import pandas as pd
import numpy as np
from pathlib import Path

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ---------------------------
# PM2.5 -> AQI (US-EPA breakpoints)
# ---------------------------
PM25_BP = [
    (0.0, 12.0, 0, 50),
    (12.1, 35.4, 51, 100),
    (35.5, 55.4, 101, 150),
    (55.5, 150.4, 151, 200),
    (150.5, 250.4, 201, 300),
    (250.5, 350.4, 301, 400),
    (350.5, 500.4, 401, 500),
]

def pm25_to_aqi(pm):
    if pm is None or (isinstance(pm, float) and math.isnan(pm)):
        return np.nan
    try:
        pm = float(pm)
    except Exception:
        return np.nan
    for lo, hi, alo, ahi in PM25_BP:
        if lo <= pm <= hi:
            return (ahi - alo) / (hi - lo) * (pm - lo) + alo
    return 500.0

def safe_float(x):
    try:
        if pd.isna(x):
            return np.nan
        return float(x)
    except Exception:
        return np.nan

# ---------------------------
# SAPID mapping
# ---------------------------
SAPID_MAP = {
    "1_Good": (25, 5),
    "2_Moderate": (75, 8),
    "3_Unhealthy_For_Sensitive_Groups": (125, 10),
    "4_Unhealthy": (175, 15),
    "5_Very_Unhealthy": (250, 20),
}

def sapid_soft_aqi(folder_name):
    if folder_name in SAPID_MAP:
        mu, sigma = SAPID_MAP[folder_name]
        rnd = np.random.RandomState(RANDOM_SEED + (hash(folder_name) % 9999))
        return float(mu + rnd.normal(0, sigma))
    # fallback: parse leading number
    try:
        num = int(folder_name.split("_")[0])
        mids = [25, 75, 125, 175, 250]
        idx = max(0, min(4, num - 1))
        return float(mids[idx])
    except Exception:
        return np.nan

# ---------------------------
# AQI bin helper
# ---------------------------
def aqi_bin(aqi_val):
    if pd.isna(aqi_val):
        return -1
    bins = [-1, 50, 100, 150, 200, 300, 500]
    for i in range(len(bins)-1):
        if bins[i] < aqi_val <= bins[i+1]:
            return i
    return len(bins)-2

# ---------------------------
# TRAQID resolver: search subfolders for filename
# ---------------------------
def resolve_traqid_image(root_dir, filename):
    """
    Recursively find a file whose basename equals filename (case-insensitive).
    Returns absolute path string or empty string if not found.
    """
    target = filename.lower()
    for rd, _, files in os.walk(root_dir):
        for f in files:
            if f.lower() == target:
                return str(Path(rd) / f)
    return ""

# ---------------------------
# Dataset processors
# ---------------------------
def process_ind_nep(ind_csv_path, all_img_dir, rows):
    try:
        df = pd.read_csv(ind_csv_path)
    except Exception as e:
        print("ERROR reading IND_NEP csv:", e)
        return
    for _, r in df.iterrows():
        fname = r.get("Filename") or r.get("filename") or ""
        if not isinstance(fname, str) or fname.strip() == "":
            continue
        img = str(Path(all_img_dir) / fname)
        aqi = safe_float(r.get("AQI"))
        pm25 = safe_float(r.get("PM2.5"))
        pm10 = safe_float(r.get("PM10"))
        aqi_cat = r.get("AQI_Class") if r.get("AQI_Class") is not None else np.nan
        hour = r.get("Hour")
        dn = "Unknown"
        if isinstance(hour, str) and ":" in hour:
            try:
                hh = int(hour.split(":")[0])
                dn = "Day" if 6 <= hh <= 18 else "Night"
            except Exception:
                pass
        has_aqi = not pd.isna(aqi)
        has_pm25 = not pd.isna(pm25)
        has_pm10 = not pd.isna(pm10)
        aqi_soft = aqi if has_aqi else (pm25_to_aqi(pm25) if has_pm25 else np.nan)
        rows.append({
            "image_path": img,
            "source": "IND_NEP",
            "aqi": aqi if has_aqi else np.nan,
            "pm25": pm25 if has_pm25 else np.nan,
            "pm10": pm10 if has_pm10 else np.nan,
            "aqi_category": aqi_cat,
            "aqi_soft": aqi_soft,
            "day_night": dn,
            "depth_path": "",
            "has_aqi": bool(has_aqi),
            "has_pm25": bool(has_pm25),
            "has_pm10": bool(has_pm10),
            "notes": ""
        })

def process_pm25vision(meta_csv_path, project_root, rows):
    try:
        df = pd.read_csv(meta_csv_path)
    except Exception as e:
        print("ERROR reading PM25VISION meta:", e)
        return
    for _, r in df.iterrows():
        rel = r.get("image_path") or r.get("filepath") or ""
        if not isinstance(rel, str) or rel.strip() == "":
            continue
        # The metadata stores relative path like: data/images/128_pm25/train/xxxx.jpg
        img = str(Path(project_root) / rel)
        pm25 = safe_float(r.get("pm25") or r.get("PM2.5"))
        aqi_val = pm25_to_aqi(pm25) if not pd.isna(pm25) else np.nan
        has_pm25 = not pd.isna(pm25)
        has_aqi = not pd.isna(aqi_val)
        rows.append({
            "image_path": img,
            "source": "PM25VISION",
            "aqi": aqi_val if has_aqi else np.nan,
            "pm25": pm25 if has_pm25 else np.nan,
            "pm10": np.nan,
            "aqi_category": np.nan,
            "aqi_soft": aqi_val,
            "day_night": "Unknown",
            "depth_path": "",
            "has_aqi": bool(has_aqi),
            "has_pm25": bool(has_pm25),
            "has_pm10": False,
            "notes": ""
        })

def process_traqid(csv_path, img_root, rows):
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print("ERROR reading TRAQID csv:", e)
        return
    missing_count = 0
    for _, r in df.iterrows():
        img_col = r.get("Image") or r.get("ImageName") or ""
        if isinstance(img_col, str) and img_col.strip() != "":
            filename = img_col
        else:
            try:
                filename = f"{int(img_col)}.jpg"
            except Exception:
                filename = ""
        resolved = ""
        if filename:
            resolved = resolve_traqid_image(img_root, filename)
        if resolved == "":
            # leave empty path (will be marked missing later)
            missing_count += 1
        aqi = safe_float(r.get("aqi"))
        pm25 = safe_float(r.get("PM2.5"))
        pm10 = safe_float(r.get("PM10"))
        dn = r.get("Day_or_Night") or r.get("day_night") or "Unknown"
        has_aqi = not pd.isna(aqi)
        has_pm25 = not pd.isna(pm25)
        has_pm10 = not pd.isna(pm10)
        aqi_soft = aqi if has_aqi else (pm25_to_aqi(pm25) if has_pm25 else np.nan)
        rows.append({
            "image_path": resolved,
            "source": "TRAQID",
            "aqi": aqi if has_aqi else np.nan,
            "pm25": pm25 if has_pm25 else np.nan,
            "pm10": pm10 if has_pm10 else np.nan,
            "aqi_category": r.get("aqi_cat") or np.nan,
            "aqi_soft": aqi_soft,
            "day_night": dn,
            "depth_path": "",
            "has_aqi": bool(has_aqi),
            "has_pm25": bool(has_pm25),
            "has_pm10": bool(has_pm10),
            "notes": "" if resolved else "traqid_not_found"
        })
    print(f"TRAQID resolver: {missing_count} filenames not resolved (will be marked).")

def process_sapid(sapid_root, rows):
    if not Path(sapid_root).exists():
        return
    for folder in os.listdir(sapid_root):
        folder_path = Path(sapid_root) / folder
        if not folder_path.is_dir():
            continue
        for root, _, files in os.walk(folder_path):
            for fn in files:
                if not fn.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                img = str(Path(root) / fn)
                soft = sapid_soft_aqi(folder)
                rows.append({
                    "image_path": img,
                    "source": "SAPID",
                    "aqi": soft,
                    "pm25": np.nan,
                    "pm10": np.nan,
                    "aqi_category": folder,
                    "aqi_soft": soft,
                    "day_night": "Unknown",
                    "depth_path": "",
                    "has_aqi": True,
                    "has_pm25": False,
                    "has_pm10": False,
                    "notes": ""
                })

# ---------------------------
# MAIN
# ---------------------------
def main(args):
    project_root = Path(args.project_root).expanduser().resolve()
    data_dir = project_root / "data"
    if not data_dir.exists():
        print("ERROR: data directory not found:", data_dir)
        sys.exit(1)

    rows = []

    # IND_NEP
    ind_csv = data_dir / "kaggle" / "Air Pollution Image Dataset" / "Air Pollution Image Dataset" / "Combined_Dataset" / "IND_and_Nep_AQI_Dataset.csv"
    ind_all_img = data_dir / "kaggle" / "Air Pollution Image Dataset" / "Air Pollution Image Dataset" / "Combined_Dataset" / "All_img"
    if ind_csv.exists():
        print("Processing IND_NEP:", ind_csv)
        process_ind_nep(str(ind_csv), str(ind_all_img), rows)
    else:
        print("IND_NEP CSV not found:", ind_csv)

    # PM25VISION (train + test)
    for meta in [data_dir / "pm25vision" / "metadata_pm25_train.csv", data_dir / "pm25vision" / "metadata_pm25_test.csv"]:
        if meta.exists():
            print("Processing PM25VISION:", meta)
            process_pm25vision(str(meta), project_root, rows)

    # TRAQID
    traqid_csv = data_dir / "TRAQID_sample" / "TRAQID.csv"
    traqid_imgs = data_dir / "TRAQID_sample" / "Images"
    if traqid_csv.exists():
        print("Processing TRAQID:", traqid_csv)
        process_traqid(str(traqid_csv), str(traqid_imgs), rows)
    else:
        print("TRAQID CSV not found:", traqid_csv)

    # SAPID
    sapid_dir = data_dir / "kaggle" / "Smartphone-Based Air Pollution Image Dataset (SAPID)" / "Smartphone-Based Air Pollution Image Dataset (SAPID)"
    if sapid_dir.exists():
        print("Processing SAPID:", sapid_dir)
        process_sapid(str(sapid_dir), rows)
    else:
        print("SAPID dir not found (skipping):", sapid_dir)

    df = pd.DataFrame(rows)
    print("Collected rows:", len(df))

    # Fill aqi_soft when pm25 available
    mask_fill = df["aqi_soft"].isna() & df["pm25"].notna()
    if mask_fill.any():
        df.loc[mask_fill, "aqi_soft"] = df.loc[mask_fill, "pm25"].apply(pm25_to_aqi)

    # aqi_bin
    df["aqi_bin"] = df["aqi_soft"].apply(lambda x: aqi_bin(x) if not pd.isna(x) else -1)

    # sample_weight per source
    counts = df["source"].value_counts().to_dict()
    df["sample_weight"] = df["source"].map(lambda s: 1.0 / math.sqrt(counts.get(s, 1)))

    # depth discovery
    depth_dir = data_dir / "images" / "128_depth"
    def find_depth_path(row):
        name = Path(row["image_path"]).name
        p1 = depth_dir / f"{name}.png"
        p2 = depth_dir / f"{name}.npy"
        if p1.exists():
            return str(p1)
        if p2.exists():
            return str(p2)
        return ""
    df["depth_path"] = df.apply(find_depth_path, axis=1)
    df["depth_available"] = df["depth_path"].apply(lambda p: bool(p and str(p).strip()))

    # ensure numeric columns and bools
    for col in ["aqi","pm25","pm10","aqi_soft"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["has_aqi"] = df["has_aqi"].fillna(False).astype(bool)
    df["has_pm25"] = df["has_pm25"].fillna(False).astype(bool)
    df["has_pm10"] = df["has_pm10"].fillna(False).astype(bool)

    # mark rows with no labels
    df["any_label"] = df[["has_aqi","has_pm25","has_pm10"]].any(axis=1)
    if (~df["any_label"]).any():
        print("WARNING: rows with no labels found. Count:", (~df["any_label"]).sum())
        df.loc[~df["any_label"], "notes"] = df["notes"].astype(str) + ";no_labels"

    out_csv = data_dir / "master_v1.csv"
    df.to_csv(out_csv, index=False)
    print("Wrote master CSV:", out_csv)

    # summary
    summary = df.groupby("source").agg(
        count=("image_path","count"),
        has_aqi=("has_aqi","sum"),
        has_pm25=("has_pm25","sum"),
        has_pm10=("has_pm10","sum"),
        depth_available=("depth_available","sum"),
    )
    print(summary)
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", required=True, help="project root path (contains data/)")
    args = parser.parse_args()
    main(args)

#!/usr/bin/env python3
import os
import pandas as pd
from PIL import Image
from tqdm import tqdm

ROOT = "data/pm25vision"

OUT_IMG_TRAIN = "data/images/128_pm25/train"
OUT_IMG_TEST  = "data/images/128_pm25/test"

META_TRAIN_OUT = "data/pm25vision/metadata_pm25_train.csv"
META_TEST_OUT  = "data/pm25vision/metadata_pm25_test.csv"

IMG_SIZE = (128, 128)

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def resize_and_copy(src, dst):
    try:
        img = Image.open(src).convert("RGB")
        img = img.resize(IMG_SIZE)
        img.save(dst)
        return True
    except:
        return False

def process_split(split):
    meta_path = f"{ROOT}/{split}/metadata.csv"
    df = pd.read_csv(meta_path)

    # Build output folders
    if split == "train":
        out_folder = OUT_IMG_TRAIN
        out_meta = META_TRAIN_OUT
    else:
        out_folder = OUT_IMG_TEST
        out_meta = META_TEST_OUT

    ensure_dir(out_folder)

    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {split}"):
        fname = row["filename"]
        pm = row["pm25"]

        src_img = f"{ROOT}/{split}/images/{fname}"
        dst_img = f"{out_folder}/{fname}"

        ok = resize_and_copy(src_img, dst_img)
        if not ok:
            continue

        rows.append({
            "image_path": dst_img,
            "pm25": float(pm),
            "lat": row.get("latitude", None),
            "lon": row.get("longitude", None),
            "station_id": row.get("station_id", None),
            "captured_at": row.get("captured_at", None),
            "quality_score": row.get("quality_score", None),
            "camera_angle": row.get("camera_angle", None),
        })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_meta, index=False)
    print(f"Saved → {out_meta}  ({len(out_df)} rows)")


def main():
    process_split("train")
    process_split("test")
    print("\nPM25Vision preparation complete.")

if __name__ == "__main__":
    main()

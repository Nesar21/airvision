#!/usr/bin/env python3
import os
import argparse
from pathlib import Path
from PIL import Image
import pandas as pd
from tqdm import tqdm
import hashlib
import datetime

def resize_image(src, dst, size=(128,128)):
    img = Image.open(src).convert("RGB")
    img = img.resize(size, Image.BILINEAR)
    img.save(dst)

def safe_float(x):
    try: return float(x)
    except: return None

def extract_label_from_path(path):
    p = str(path).lower()
    if "good" in p: return 50
    if "moderate" in p: return 100
    if "unhealthy_for_sensitive" in p: return 150
    if "unhealthy" in p and "for" not in p: return 200
    if "very_unhealthy" in p: return 300
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    parser.add_argument("--out_raw", required=True)
    parser.add_argument("--out_resized", required=True)
    parser.add_argument("--metadata", required=True)
    args = parser.parse_args()

    src = Path(args.src)
    out_raw = Path(args.out_raw)
    out_resized = Path(args.out_resized)
    metadata_path = Path(args.metadata)

    out_raw.mkdir(parents=True, exist_ok=True)
    out_resized.mkdir(parents=True, exist_ok=True)

    if metadata_path.exists():
        df = pd.read_csv(metadata_path)
    else:
        df = pd.DataFrame(columns=[
            "image_path","lat","lon","city","country","timestamp",
            "aqi","pm25","pm10","temp","humidity","wind",
            "satellite_path","news","source_dataset","orig_path"
        ])

    used = set(df["orig_path"].astype(str).tolist())

    all_files = list(src.rglob("*.*"))
    total = sum(1 for f in all_files if f.suffix.lower() in [".jpg",".jpeg",".png"])

    print(f"Total image files detected: {total}")

    new_rows = []
    for file in tqdm(all_files, desc="processing"):
        if file.suffix.lower() not in [".jpg",".jpeg",".png"]:
            continue

        orig = str(file.resolve())
        if orig in used:
            continue

        try:
            h = hashlib.md5(orig.encode()).hexdigest()[:8]
            label = extract_label_from_path(file)
            ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            new_name = f"{file.stem}_{ts}_{h}.jpg".replace(" ","_")

            raw_dst = out_raw / new_name
            resized_dst = out_resized / new_name

            raw_dst.parent.mkdir(parents=True, exist_ok=True)
            resized_dst.parent.mkdir(parents=True, exist_ok=True)

            Image.open(file).convert("RGB").save(raw_dst)
            resize_image(raw_dst, resized_dst)

            row = {
                "image_path": str(resized_dst).replace("\\","/"),
                "lat": None,
                "lon": None,
                "city": None,
                "country": None,
                "timestamp": None,
                "aqi": label,
                "pm25": None,
                "pm10": None,
                "temp": None,
                "humidity": None,
                "wind": None,
                "satellite_path": None,
                "news": None,
                "source_dataset": file.parts[-3] if len(file.parts)>=3 else "unknown",
                "orig_path": orig
            }

            new_rows.append(row)
            used.add(orig)

        except Exception as e:
            print("ERROR:", file, e)

    if new_rows:
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
        df.to_csv(metadata_path, index=False)
        print(f"Added {len(new_rows)} new images.")
    else:
        print("No new images added.")

if __name__ == "__main__":
    main()

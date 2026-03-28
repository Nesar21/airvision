#!/usr/bin/env python3
"""
Rebuild clean 5-fold IND-NEP splits for Phase-5 Stage-3.

Steps:
1. Load metadata_image_only.csv
2. Extract city from Location ("City, Country")
3. Build AQI bins
4. Build strata = city × aqi_bin
5. Replace rare strata with "OTHER"
6. StratifiedKFold → 5 folds
7. Output: splits/fold{k}_train.csv and splits/fold{k}_val.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold

ROOT = Path(".")
meta_path = ROOT / "metadata_image_only.csv"
splits_root = ROOT / "splits"
splits_root.mkdir(exist_ok=True)

# =========================================================
# 1. Load metadata
# =========================================================

df = pd.read_csv(meta_path)

required = {
    "image_id",
    "Location",
    "image_path",
    "AQI",
    "AQI_Class",
    "label_confidence",
}

missing = required - set(df.columns)
if missing:
    raise KeyError(f"Missing columns in metadata_image_only.csv → {missing}")

# =========================================================
# 2. Extract city
# =========================================================
# Location is like: "Biratnagar, Nepal"
df["city"] = df["Location"].astype(str).apply(lambda x: x.split(",")[0].strip())

# =========================================================
# 3. Build AQI bins
# =========================================================

bins = [0, 50, 100, 150, 200, 300, 10000]
df["AQI"] = pd.to_numeric(df["AQI"], errors="coerce")
df["aqi_bin"] = pd.cut(df["AQI"], bins=bins, labels=False, include_lowest=True)

# =========================================================
# 4. Build strata = city × AQI_bin
# =========================================================

df["strata"] = df["city"].astype(str) + "_" + df["aqi_bin"].astype(str)

# =========================================================
# 5. Replace rare strata (< 5 samples)
# =========================================================

counts = df["strata"].value_counts()
rare_strata = counts[counts < 5].index

df.loc[df["strata"].isin(rare_strata), "strata"] = "OTHER"

# =========================================================
# 6. Perform StratifiedKFold
# =========================================================

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(kf.split(df, df["strata"])):
    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_val = df.iloc[val_idx].reset_index(drop=True)

    out_train = splits_root / f"fold{fold}_train.csv"
    out_val = splits_root / f"fold{fold}_val.csv"

    df_train.to_csv(out_train, index=False)
    df_val.to_csv(out_val, index=False)

    print(f"Fold {fold}: train={len(df_train)}, val={len(df_val)}  |  saved.")

print("\nDONE ✔ All city-stratified folds built successfully.")

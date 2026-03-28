#!/usr/bin/env python3
"""
STEP 3 — Strict Schema Normalization for master_v1.csv
Output: master_v1_clean.csv

Responsibilities:
  - enforce strict column order
  - cast all numeric columns
  - ensure bool masks are valid
  - ensure pm10 only exists where expected
  - verify image_path uniqueness
  - drop duplicates
  - check label consistency
  - fill missing aqi_soft
  - auto-generate aqi_bin
"""

import argparse
import pandas as pd
import numpy as np
import math
from pathlib import Path

# -------------------------------------------------------------
# PM2.5 → AQI conversion
# -------------------------------------------------------------
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
    if pm is None or pd.isna(pm):
        return np.nan
    pm = float(pm)
    for lo, hi, a_lo, a_hi in PM25_BP:
        if lo <= pm <= hi:
            return (a_hi - a_lo) / (hi - lo) * (pm - lo) + a_lo
    return 500.0

def aqi_bin(aqi):
    if pd.isna(aqi):
        return -1
    b = [-1,50,100,150,200,300,500]
    for i in range(len(b)-1):
        if b[i] < aqi <= b[i+1]:
            return i
    return 5


# -------------------------------------------------------------
# MAIN
# -------------------------------------------------------------
def main(args):
    df = pd.read_csv(args.infile)

    print("Loaded:", len(df))

    # ---------------------------------------------------------
    # STEP 1: Enforce exact column set
    # ---------------------------------------------------------
    required_cols = [
        "image_path", "source",
        "aqi", "pm25", "pm10",
        "aqi_category", "aqi_soft",
        "day_night", "depth_path",
        "has_aqi", "has_pm25", "has_pm10",
        "notes", "aqi_bin",
        "sample_weight", "depth_available"
    ]

    # add missing columns
    for col in required_cols:
        if col not in df.columns:
            df[col] = np.nan

    # prune unknown columns
    df = df[required_cols]

    # ---------------------------------------------------------
    # STEP 2: Type Casting
    # ---------------------------------------------------------
    numeric_cols = ["aqi", "pm25", "pm10", "aqi_soft", "sample_weight"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["has_aqi"] = df["has_aqi"].astype(bool)
    df["has_pm25"] = df["has_pm25"].astype(bool)
    df["has_pm10"] = df["has_pm10"].astype(bool)
    df["depth_available"] = df["depth_available"].astype(bool)

    df["image_path"] = df["image_path"].astype(str)
    df["source"] = df["source"].astype(str)
    df["day_night"] = df["day_night"].astype(str)
    df["notes"] = df["notes"].astype(str)

    # ---------------------------------------------------------
    # STEP 3: Fix aqi_soft if missing but PM25 exists
    # ---------------------------------------------------------
    mask = df["aqi_soft"].isna() & df["pm25"].notna()
    df.loc[mask, "aqi_soft"] = df.loc[mask]["pm25"].apply(pm25_to_aqi)

    # ---------------------------------------------------------
    # STEP 4: Recompute aqi_bin cleanly
    # ---------------------------------------------------------
    df["aqi_bin"] = df["aqi_soft"].apply(aqi_bin)

    # ---------------------------------------------------------
    # STEP 5: Remove duplicates
    # ---------------------------------------------------------
    before = len(df)
    df.drop_duplicates(subset=["image_path"], inplace=True)
    after = len(df)
    print(f"Removed duplicates: {before - after}")

    # ---------------------------------------------------------
    # STEP 6: Label consistency checks
    # ---------------------------------------------------------

    # pm10 should not exist outside IND_NEP + TRAQID
    wrong_pm10 = df[(df["pm10"].notna()) & (~df["source"].isin(["IND_NEP", "TRAQID"]))]
    if len(wrong_pm10) > 0:
        print("WARNING: pm10 found in unexpected rows:", len(wrong_pm10))
        df.loc[wrong_pm10.index, "pm10"] = np.nan

    # has flags must mirror numeric columns
    df["has_aqi"] = df["aqi"].notna()
    df["has_pm25"] = df["pm25"].notna()
    df["has_pm10"] = df["pm10"].notna()

    # ---------------------------------------------------------
    # STEP 7: Final cleaning
    # ---------------------------------------------------------
    df["aqi_category"] = df["aqi_category"].replace({np.nan: ""})

    # ---------------------------------------------------------
    # STEP 8: Save clean CSV
    # ---------------------------------------------------------
    out_csv = Path(args.outfile)
    df.to_csv(out_csv, index=False)
    print("Wrote clean CSV:", out_csv)

    # ---------------------------------------------------------
    # STEP 9: Print summary
    # ---------------------------------------------------------
    print(df.groupby("source").agg(
        count=("image_path", "count"),
        aqi_available=("has_aqi", "sum"),
        pm25_available=("has_pm25", "sum"),
        pm10_available=("has_pm10", "sum"),
    ))

    print("Final rows:", len(df))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="infile", required=True)
    parser.add_argument("--out", dest="outfile", required=True)
    main(parser.parse_args())

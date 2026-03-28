#!/usr/bin/env python3
import argparse
import os

import numpy as np
import pandas as pd


def build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--metadata",
        default="data/metadata_enriched.csv",
        help="Input metadata CSV with aqi_continuous, sat_brightness, sat_blur, timestamp",
    )
    ap.add_argument(
        "--out",
        default="data/metadata_scene_labels.csv",
        help="Output CSV with scene_type, scene_label, is_night",
    )
    return ap


def main():
    args = build_argparser().parse_args()

    in_path = args.metadata
    out_path = args.out

    print(f"Loading: {in_path}")
    df = pd.read_csv(in_path)

    required_cols = ["aqi_continuous", "sat_brightness", "sat_blur", "timestamp"]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Required column '{c}' is missing in {in_path}")

    # Parse timestamp → datetime
    df["timestamp_dt"] = pd.to_datetime(df["timestamp"], errors="coerce")
    valid_mask = (
        df["aqi_continuous"].notna()
        & df["sat_brightness"].notna()
        & df["sat_blur"].notna()
        & df["timestamp_dt"].notna()
    )

    print(f"Total rows: {len(df)}")
    print(f"Rows with full info (used for labeling): {valid_mask.sum()}")

    # Initialize outputs
    df["scene_type"] = "unknown"
    df["scene_label"] = -1
    df["is_night"] = 0

    # Compute adaptive thresholds from valid rows
    sub = df[valid_mask]

    bright_q30 = sub["sat_brightness"].quantile(0.30)
    bright_q60 = sub["sat_brightness"].quantile(0.60)
    blur_q70 = sub["sat_blur"].quantile(0.70)

    print("\nAdaptive thresholds (from data):")
    print(f"  bright_q30 = {bright_q30:.4f}")
    print(f"  bright_q60 = {bright_q60:.4f}")
    print(f"  blur_q70   = {blur_q70:.4f}")

    # NOTE: AQI thresholds (can be tuned)
    AQI_HAZY = 100.0
    AQI_SMOG = 200.0

    # Label mapping:
    # 0 = day_clear
    # 1 = hazy_day
    # 2 = smog_day
    # 3 = night_clear
    # 4 = night_hazy

    for idx in np.where(valid_mask)[0]:
        row = df.iloc[idx]

        aqi = float(row["aqi_continuous"])
        bright = float(row["sat_brightness"])
        blur = float(row["sat_blur"])
        ts = row["timestamp_dt"]

        hour = ts.hour
        is_night = int(hour < 6 or hour >= 19)

        if is_night:
            # Night images
            df.at[idx, "is_night"] = 1

            # Very dark or very blurred → night_hazy
            if (bright <= bright_q30) or (blur >= blur_q70):
                df.at[idx, "scene_type"] = "night_hazy"
                df.at[idx, "scene_label"] = 4
            else:
                df.at[idx, "scene_type"] = "night_clear"
                df.at[idx, "scene_label"] = 3
        else:
            # Daytime images
            df.at[idx, "is_night"] = 0

            if aqi >= AQI_SMOG:
                df.at[idx, "scene_type"] = "smog_day"
                df.at[idx, "scene_label"] = 2
            elif (aqi >= AQI_HAZY) or (blur >= blur_q70 and bright <= bright_q60):
                # either high AQI OR visually hazy
                df.at[idx, "scene_type"] = "hazy_day"
                df.at[idx, "scene_label"] = 1
            else:
                df.at[idx, "scene_type"] = "day_clear"
                df.at[idx, "scene_label"] = 0

    # Save
    df.to_csv(out_path, index=False)
    print("\nSaved:", out_path)

    print("\nScene distribution (all rows):")
    print(df["scene_type"].value_counts(dropna=False))

    print("\nScene distribution (valid labeled rows only):")
    print(df.loc[df["scene_label"] >= 0, "scene_type"].value_counts(dropna=False))


if __name__ == "__main__":
    main()

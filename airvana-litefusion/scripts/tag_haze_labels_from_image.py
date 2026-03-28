#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd


def build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--metadata",
        default="data/metadata_haze_features.csv",
        help="Input CSV with aqi_continuous, img_brightness, img_lap_var",
    )
    ap.add_argument(
        "--out",
        default="data/metadata_scene_haze.csv",
        help="Output CSV with haze_label and haze_type",
    )
    return ap


def main():
    args = build_argparser().parse_args()

    df = pd.read_csv(args.metadata)

    required_cols = ["aqi_continuous", "img_brightness", "img_lap_var"]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Required column '{c}' missing in {args.metadata}")

    # Only rows with full info
    mask = (
        df["aqi_continuous"].notna()
        & df["img_brightness"].notna()
        & df["img_lap_var"].notna()
    )
    sub = df[mask].copy()
    print("Total rows:", len(df))
    print("Rows with full info:", len(sub))

    if len(sub) == 0:
        raise RuntimeError("No rows with full AQI + haze features; cannot label.")

    # Adaptive thresholds (Balanced – option B)
    b_q40 = sub["img_brightness"].quantile(0.40)
    b_q60 = sub["img_brightness"].quantile(0.60)
    lap_q40 = sub["img_lap_var"].quantile(0.40)

    print("\nAdaptive thresholds:")
    print(f"  brightness_q40 = {b_q40:.4f}")
    print(f"  brightness_q60 = {b_q60:.4f}")
    print(f"  lap_var_q40    = {lap_q40:.4f}")

    # AQI thresholds
    AQI_CLEAR_MAX = 80.0
    AQI_HAZY_MIN = 120.0

    haze_label = np.full(len(df), -1, dtype=int)
    haze_type = np.array(["unknown"] * len(df), dtype=object)

    for idx in sub.index:
        aqi = float(df.at[idx, "aqi_continuous"])
        br = float(df.at[idx, "img_brightness"])
        lv = float(df.at[idx, "img_lap_var"])

        # HAZY/SMOG condition (Balanced B):
        # - strong AQI smog OR
        # - darker + blurrier than typical
        is_hazy = (aqi >= AQI_HAZY_MIN) or (
            (br <= b_q40) and (lv <= lap_q40)
        )

        # CLEAR condition:
        is_clear = (aqi <= AQI_CLEAR_MAX) and (br >= b_q60) and (lv >= lap_q40)

        if is_hazy and not is_clear:
            haze_label[idx] = 1
            haze_type[idx] = "hazy_smog"
        elif is_clear and not is_hazy:
            haze_label[idx] = 0
            haze_type[idx] = "clear"
        else:
            # ambiguous → keep as unknown
            pass

    df["haze_label"] = haze_label
    df["haze_type"] = haze_type

    # Save
    df.to_csv(args.out, index=False)
    print("\nSaved:", args.out)

    # Show distribution
    print("\nHaze label distribution (0=clear,1=hazy,-1=unknown):")
    print(df["haze_label"].value_counts(dropna=False))


if __name__ == "__main__":
    main()

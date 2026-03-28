#!/usr/bin/env python3
import pandas as pd
import numpy as np
import argparse
import os

# -----------------------------------------
# CLASS → RANGE MAPPING (expert-designed)
# -----------------------------------------
AQI_MAP = {
    50:  (35, 65),      # Good
    100: (80, 120),     # Moderate
    150: (130, 170),    # Unhealthy Sensitive
    200: (180, 230),    # Unhealthy
    250: (230, 280),    # Very Unhealthy
    300: (280, 350),    # Hazardous 1
    350: (320, 400),    # Hazardous 2
    400: (350, 450),    # Severe
    450: (400, 500),    # Critical
    500: (450, 500)     # Max
}

def map_to_continuous(aqi_class):
    try:
        class_val = int(float(aqi_class))
    except:
        return np.nan

    if class_val not in AQI_MAP:
        return np.nan

    lo, hi = AQI_MAP[class_val]
    return np.random.uniform(lo, hi)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.metadata)

    if "aqi" not in df.columns:
        raise ValueError("metadata must contain 'aqi' column.")

    df["aqi"] = df["aqi"].astype(str).str.replace(".0", "", regex=False)

    df["aqi_continuous"] = df["aqi"].apply(map_to_continuous)

    df.to_csv(args.out, index=False)
    print(f"Saved enriched metadata → {args.out}")
    print(df.head())

if __name__ == "__main__":
    main()

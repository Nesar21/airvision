#!/usr/bin/env python3
"""
Compute overall AQI index from Kaggle pollutant-wise AQI data.

Input:
    data/AQI_wide.csv  (pivoted by city,state,timestamp, pollutant_id→columns)

Output:
    data/AQI_wide_with_aqi.csv  (adds 'aqi_overall' column)
"""

import os
import pandas as pd
import numpy as np

IN_PATH = "data/AQI_wide.csv"
OUT_PATH = "data/AQI_wide_with_aqi.csv"

def main():
    if not os.path.exists(IN_PATH):
        raise FileNotFoundError(f"Input file not found: {IN_PATH}")

    df = pd.read_csv(IN_PATH)

    # pollutant columns we expect (keep only those that exist)
    pollutant_cols = [
        "PM2.5",
        "PM10",
        "NO2",
        "SO2",
        "CO",
        "OZONE",
        "NH3",
    ]
    pollutant_cols = [c for c in pollutant_cols if c in df.columns]

    if not pollutant_cols:
        raise ValueError("No pollutant columns found in AQI_wide.csv")

    # Compute overall AQI as max of available pollutant indices for each row
    df["aqi_overall"] = df[pollutant_cols].max(axis=1, skipna=True)

    # Drop rows where overall AQI is NaN (no pollutants at all)
    before = len(df)
    df = df[~df["aqi_overall"].isna()].reset_index(drop=True)
    after = len(df)

    print(f"Rows before: {before}, rows after dropping NaN AQI: {after}")
    print("Using pollutant columns:", pollutant_cols)

    df.to_csv(OUT_PATH, index=False)
    print(f"Saved with AQI label → {OUT_PATH}")
    print(df.head())


if __name__ == "__main__":
    main()

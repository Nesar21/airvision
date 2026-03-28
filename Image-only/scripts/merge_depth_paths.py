#!/usr/bin/env python3
"""
STEP 5 — Merge MiDaS depth maps into metadata
Input:
    data/master_v1_clean.csv
    data/images/128_depth/*.npy

Output:
    data/master_v2.csv

Adds:
    - depth_path
    - depth_available
    - consistent ordering
"""

import argparse
import pandas as pd
from pathlib import Path
import os

def main(args):
    in_csv = Path(args.in_csv).resolve()
    out_csv = Path(args.out_csv).resolve()
    depth_dir = Path(args.depth_dir).resolve()

    print("Loading:", in_csv)
    df = pd.read_csv(in_csv)
    print("Rows:", len(df))

    # build depth lookup
    depth_map = {}
    print("Indexing depth files…")
    for p in depth_dir.glob("*.npy"):
        depth_map[p.name] = str(p)

    def resolve_depth(row):
        bn = Path(row["image_path"]).name
        fname = f"{bn}.npy"
        return depth_map.get(fname, "")

    df["depth_path"] = df.apply(resolve_depth, axis=1)
    df["depth_available"] = df["depth_path"].apply(lambda x: x != "")

    # REPORT
    print(df["depth_available"].value_counts())
    print("\nDepth availability per source:")
    print(df.groupby("source")["depth_available"].sum())

    # enforce ordered schema
    cols = [
        "image_path",
        "source",
        "aqi",
        "pm25",
        "pm10",
        "aqi_soft",
        "aqi_category",
        "aqi_bin",
        "day_night",
        "depth_path",
        "depth_available",
        "has_aqi",
        "has_pm25",
        "has_pm10",
        "sample_weight",
        "notes",
    ]

    # keep extra columns if exist
    final_cols = [c for c in cols if c in df.columns] + \
                 [c for c in df.columns if c not in cols]

    df = df[final_cols]

    print("Writing:", out_csv)
    df.to_csv(out_csv, index=False)

    print("Done. Final rows:", len(df))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_csv", required=True)
    parser.add_argument("--depth_dir", required=True)
    parser.add_argument("--out_csv", required=True)
    main(parser.parse_args())

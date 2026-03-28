#!/usr/bin/env python3
"""
make_splits_with_night.py
Ensure TRAQID nights are present in train.
Usage:
  python scripts/make_splits_with_night.py --csv data/master_v2.csv --out_dir splits --traqid_train_frac 0.6 --seed 42
"""
import argparse, os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

def stratified_split(df, strat_col, test_size, seed):
    # simple stratified split on strat_col
    train, test = train_test_split(df, test_size=test_size, random_state=seed, stratify=df[strat_col])
    return train.reset_index(drop=True), test.reset_index(drop=True)

def main(args):
    df = pd.read_csv(args.csv)
    seed = args.seed

    # separate TRAQID and NON-TRAQID
    traqid = df[df['source']=='TRAQID'].copy()
    others = df[df['source']!='TRAQID'].copy()

    # desired fractions for TRAQID
    t_train_frac = args.traqid_train_frac
    t_val_frac   = args.traqid_val_frac
    t_test_frac  = args.traqid_test_frac
    assert abs((t_train_frac + t_val_frac + t_test_frac) - 1.0) < 1e-6

    # split TRAQID by day_night stratification if available
    if 'day_night' in traqid.columns:
        strat = traqid['day_night'].fillna('Unknown')
    else:
        strat = None

    # compute counts
    n = len(traqid)
    if n>0:
        # first split train vs rest
        train_t, rest_t = train_test_split(traqid, train_size=t_train_frac, random_state=seed, stratify=strat)
        # now split rest into val/test proportionally
        if len(rest_t)>0:
            # scale ratios
            val_ratio = t_val_frac / (t_val_frac + t_test_frac)
            val_t, test_t = train_test_split(rest_t, train_size=val_ratio, random_state=seed, stratify=rest_t['day_night'].fillna('Unknown') if 'day_night' in rest_t.columns else None)
        else:
            val_t = pd.DataFrame(columns=traqid.columns)
            test_t = pd.DataFrame(columns=traqid.columns)
    else:
        train_t = val_t = test_t = pd.DataFrame(columns=traqid.columns)

    # split others: keep previous logic (train heavy)
    # default others split: 90% train, 10% val, test small (we keep test small, e.g., 0)
    others_train, others_val = train_test_split(others, test_size=0.10, random_state=seed, stratify=others['aqi_bin'] if 'aqi_bin' in others.columns else None)
    # combine
    train_df = pd.concat([others_train, train_t]).sample(frac=1, random_state=seed).reset_index(drop=True)
    val_df   = pd.concat([others_val, val_t]).sample(frac=1, random_state=seed).reset_index(drop=True)
    test_df  = test_t.sample(frac=1, random_state=seed).reset_index(drop=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(out_dir/'train.csv', index=False)
    val_df.to_csv(out_dir/'val.csv', index=False)
    test_df.to_csv(out_dir/'test.csv', index=False)

    print("Counts:")
    print("TRAIN", len(train_df), "VAL", len(val_df), "TEST", len(test_df))
    print("TRAQID in TRAIN:", train_df['source'].eq('TRAQID').sum())
    print("TRAQID in VAL:", val_df['source'].eq('TRAQID').sum())
    print("TRAQID in TEST:", test_df['source'].eq('TRAQID').sum())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True)
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--traqid_train_frac', type=float, default=0.6)
    parser.add_argument('--traqid_val_frac', type=float, default=0.2)
    parser.add_argument('--traqid_test_frac', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    main(args)

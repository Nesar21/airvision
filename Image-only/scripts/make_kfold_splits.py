# scripts/make_kfold_splits.py
# Run:
#   python scripts/make_kfold_splits.py \
#       --csv data/master_v2.csv \
#       --out_dir splits \
#       --k 4

import argparse
import os
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold

def make_folds(df, k, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    # Build a robust stratification label
    def bin_pm25(x):
        if pd.isna(x): 
            return 0
        if x < 50: return 1
        if x < 100: return 2
        if x < 200: return 3
        if x < 300: return 4
        return 5

    df["pm25_bin"] = df["pm25"].apply(bin_pm25)
    df["strat"] = (
        df["pm25_bin"].astype(str) + "_" +
        df["day_night"].fillna("U") + "_" +
        df["source"].fillna("U")
    )

    skf = StratifiedKFold(
        n_splits=k, shuffle=True, random_state=42
    )

    for fold, (train_idx, val_idx) in enumerate(
        skf.split(df, df["strat"])
    ):
        fold_dir = Path(out_dir) / f"fold{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        df.iloc[train_idx].to_csv(fold_dir/"train.csv", index=False)
        df.iloc[val_idx].to_csv(fold_dir/"val.csv", index=False)

        print(f"[Fold {fold}] train={len(train_idx)}  val={len(val_idx)}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--k", type=int, default=4)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    make_folds(df, args.k, args.out_dir)

if __name__ == "__main__":
    main()

# make_kfold_splits.py
# Generates 4-fold train/val splits.
#
# Usage:
#   python make_kfold_splits.py --master data/master_v2.csv --outdir splits
#
# Output structure:
#   splits/
#       fold0/train.csv  fold0/val.csv
#       fold1/train.csv  fold1/val.csv
#       fold2/train.csv  fold2/val.csv
#       fold3/train.csv  fold3/val.csv

import argparse
import pandas as pd
from pathlib import Path
from sklearn.model_selection import KFold

def make_kfold_splits(master_csv, outdir, k=4, seed=42):
    df = pd.read_csv(master_csv)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    kf = KFold(n_splits=k, shuffle=True, random_state=seed)

    for fold, (train_idx, val_idx) in enumerate(kf.split(df)):
        fold_dir = outdir / f"fold{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        df.iloc[train_idx].to_csv(fold_dir/"train.csv", index=False)
        df.iloc[val_idx].to_csv(fold_dir/"val.csv", index=False)

        print(f"[OK] fold {fold} → {fold_dir}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", type=str, required=True)
    ap.add_argument("--outdir", type=str, required=True)
    args = ap.parse_args()

    make_kfold_splits(args.master, args.outdir)

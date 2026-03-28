# make_splits.py — DAY-ONLY SAFE VERSION
# Works even when TRAQID = 0

import argparse
import pandas as pd
import os
from pathlib import Path
from sklearn.model_selection import train_test_split

def main(args):
    df = pd.read_csv(args.csv)
    print("Loaded:", len(df))

    # Identify TRAQID rows
    traqid = df[df["source"] == "TRAQID"]
    others = df[df["source"] != "TRAQID"]

    print("TRAQID rows:", len(traqid))
    print("NON-TRAQID rows:", len(others))

    # Case 1: TRAQID exists → split normally
    if len(traqid) > 0:
        traq_train, traq_test = train_test_split(
            traqid,
            test_size=0.2,
            random_state=42,
            stratify=traqid["aqi_bin"],
        )
        print("TRAQID train/test:", len(traq_train), len(traq_test))
    else:
        # Case 2: TRAQID = 0 → skip
        traq_train, traq_test = pd.DataFrame(), pd.DataFrame()
        print("No TRAQID present → skipping TRAQID split.")

    # Split the other datasets
    train_df, test_df = train_test_split(
        others,
        test_size=0.2,
        random_state=42,
        stratify=others["aqi_bin"],
    )

    # Add TRAQID test rows only if exist
    full_test = pd.concat([test_df, traq_test], ignore_index=True)

    # Now create validation split from train
    train_df, val_df = train_test_split(
        train_df,
        test_size=0.12,  # 12% of training → 70/18/12 final
        random_state=42,
        stratify=train_df["aqi_bin"],
    )

    out = Path(args.out_dir)
    out.mkdir(exist_ok=True, parents=True)

    train_df.to_csv(out / "train.csv", index=False)
    val_df.to_csv(out / "val.csv", index=False)
    full_test.to_csv(out / "test.csv", index=False)

    print("\nSaved:")
    print(" train:", len(train_df))
    print(" val:", len(val_df))
    print(" test:", len(full_test))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()
    main(args)

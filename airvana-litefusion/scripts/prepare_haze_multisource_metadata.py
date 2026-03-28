#!/usr/bin/env python3
import os
import argparse
import pandas as pd

IMG_ROOT = "data"  # everything is inside ./data

def list_with_label(root, label):
    rows = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith((".jpg", ".jpeg", ".png")):
                rows.append({
                    "image_path": os.path.join(dirpath, fn),
                    "haze_label": label,
                    "source": root
                })
    return rows

def build(args):
    rows = []

    # 1) TRAQID night images (mostly hazy/smog)
    #    You can fine-tune this mapping later if needed.
    traqid_front = "data/TRAQID_sample/Images/2/Front"
    traqid_rear = "data/TRAQID_sample/Images/2/Rear"
    rows += list_with_label(traqid_front, 1)   # hazy/smog
    rows += list_with_label(traqid_rear, 1)

    # 2) SAPID (smartphone AQI images)
    sapid_root = "data/kaggle/Smartphone-Based Air Pollution Image Dataset (SAPID)/Smartphone-Based Air Pollution Image Dataset (SAPID)"
    sapid_mapping = {
        "1_Good": 0,   # clear
        "2_Moderate": 0,  # still mostly clear
        "3_Unhealthy_For_Sensitive_Groups": 1,  # hazy/smog
        "4_Unhealthy": 1,
        "5_Very_Unhealthy": 1,
    }
    for sub, label in sapid_mapping.items():
        root = os.path.join(sapid_root, sub)
        if os.path.isdir(root):
            rows += list_with_label(root, label)

    # 3) Air Pollution Image Dataset – IND_and_NEP
    ind_nep_root = "data/kaggle/Air Pollution Image Dataset/Air Pollution Image Dataset/Combined_Dataset/IND_and_NEP"
    ind_mapping = {
        "a_Good": 0,
        "b_Moderate": 0,
        "c_Unhealthy_for_Sensitive_Groups": 1,
        "d_Unhealthy": 1,
        "e_Very_Unhealthy": 1,
        "f_Severe": 1,
    }
    for sub, label in ind_mapping.items():
        root = os.path.join(ind_nep_root, sub)
        if os.path.isdir(root):
            rows += list_with_label(root, label)

    # 4) pm25vision – use train + test
    for split in ["train", "test"]:
        root = f"data/pm25vision/{split}/images"
        if os.path.isdir(root):
            # pm25vision already has metadata; we only need paths here.
            # Treat all as hazy/smog (these are high PM2.5 events).
            rows += list_with_label(root, 1)

    # 5) Your global dataset (37k) – use our fixed labels as extra data
    base_csv = "data/metadata_haze_labels_fixed.csv"
    if os.path.exists(base_csv):
        df_base = pd.read_csv(base_csv)
        df_base = df_base[df_base["haze_label"] >= 0].copy()
        df_base["source"] = "metadata_haze_fixed"
        rows += df_base.to_dict(orient="records")

    df = pd.DataFrame(rows)

    # Normalise paths relative to repo root
    df["image_path"] = df["image_path"].apply(os.path.normpath)

    # Simple sanity checks
    print("Total rows:", len(df))
    print("Label distribution (0=clear,1=hazy/smog):")
    print(df["haze_label"].value_counts())

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print("Saved:", args.out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/metadata_haze_multisource.csv")
    args = ap.parse_args()
    build(args)

if __name__ == "__main__":
    main()

# src/splits/phase3_split.py

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold

from src.utils.env_utils import init_env
from src.utils.manifest import log_run

DATA_ROOT = Path(".")
META_CSV = DATA_ROOT / "metadata_image_only.csv"
INDEX_CSV = DATA_ROOT / "phase2_index_image_only.csv"

HOLDOUT_CITIES = ["Delhi", "Mumbai", "Kanpur"]
N_SPLITS = 5
AQI_BINS = [0, 50, 100, 150, 200, 300, 10000]

def bin_aqi(aqi):
    return np.digitize(aqi, AQI_BINS, right=False)

def build_splits():
    init_env(anomaly_for_debug=False)

    df_meta = pd.read_csv(META_CSV)
    df_idx  = pd.read_csv(INDEX_CSV)

    # After merge, columns become _x and _y
    df = df_meta.merge(df_idx, on="image_id")

    # -----------------------------------------------------
    # USE ONLY *_x COLUMNS (metadata is authoritative)
    # -----------------------------------------------------
    df["city"] = df["Location_x"].apply(lambda x: x.split(",")[0].strip())

    # -----------------------------------------------------
    # Holdout set by pure city name
    # -----------------------------------------------------
    mask_holdout = df["city"].isin(HOLDOUT_CITIES)
    df_holdout = df[mask_holdout].reset_index(drop=True)
    df_train_full = df[~mask_holdout].reset_index(drop=True)

    # -----------------------------------------------------
    # IND-NEP only (correct column)
    # -----------------------------------------------------
    df_ind_nep = df_train_full[df_train_full["country_group_x"] == "IND_NEP"].copy()

    # -----------------------------------------------------
    # AQI binning (correct column)
    # -----------------------------------------------------
    df_ind_nep["aqi_bin"] = df_ind_nep["AQI_x"].apply(bin_aqi)

    # -----------------------------------------------------
    # Group by pure city name
    # -----------------------------------------------------
    groups = df_ind_nep["city"].values
    y      = df_ind_nep["aqi_bin"].values

    sgkf = StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=42
    )

    split_dir = Path("splits")
    split_dir.mkdir(exist_ok=True)

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(df_ind_nep, y, groups)):
        df_ind_nep.iloc[train_idx].to_csv(split_dir / f"fold{fold}_train.csv", index=False)
        df_ind_nep.iloc[val_idx].to_csv(split_dir / f"fold{fold}_val.csv", index=False)

    # -----------------------------------------------------
    # Save holdout cities
    # -----------------------------------------------------
    df_holdout.to_csv("holdout_3city.csv", index=False)

    # -----------------------------------------------------
    # Log summary
    # -----------------------------------------------------
    log_run(
        phase="phase3_split",
        stage="cv_generation",
        description="3-city holdout + 5-fold SGKF using *_x metadata columns",
        config={"holdout_cities": HOLDOUT_CITIES, "n_splits": N_SPLITS},
        metrics={"holdout_rows": len(df_holdout), "train_rows": len(df_ind_nep)}
    )

    print("PHASE 3 COMPLETE")
    print("Holdout saved → holdout_3city.csv")
    print("Folds saved → splits/fold*_train.csv, splits/fold*_val.csv")


if __name__ == "__main__":
    build_splits()

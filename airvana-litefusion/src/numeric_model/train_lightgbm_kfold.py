#!/usr/bin/env python3
"""
train_lightgbm_kfold.py

Train a LightGBM AQI regression model with 5-fold CV using:
  - weather API features (*_fetch)
  - satellite features (sat_*)

Label:
  - aqi_continuous

Outputs:
  - Fold models: models/lightgbm/weather_fold{0..4}.txt
  - CV metrics: results/lightgbm/lightgbm_cv_metrics.txt
  - Feature importance per fold:
      results/lightgbm/feature_importance_fold{0..4}.csv
"""

import os
import argparse
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import lightgbm as lgb


# ---------------------------
# Feature builder
# ---------------------------

def get_features(df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    """
    Build feature matrix X from *_fetch and satellite columns.
    """

    candidate_cols = [
        "pm25_fetch",
        "pm10_fetch",
        "temp_fetch",
        "humidity_fetch",
        "pressure_fetch",
        "visibility_fetch",
        "wind_fetch",
        "wind_deg_fetch",
        "feels_like_fetch",
        "dew_point_fetch",
        "sat_brightness",
        "sat_blur",
        "sat_color_skew",
    ]

    cols = [c for c in candidate_cols if c in df.columns]
    if not cols:
        raise ValueError("No numeric feature columns found for LightGBM.")

    X = df[cols].astype(float)
    # simple median imputation
    X = X.fillna(X.median(numeric_only=True))

    return X.values, cols


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mse = mean_squared_error(y_true, y_pred)
    return float(np.sqrt(mse))


# ---------------------------
# Main training
# ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--metadata",
        required=True,
        help="Path to metadata CSV (e.g. data/metadata_sat_features.csv)",
    )
    ap.add_argument(
        "--out_dir",
        required=True,
        help="Directory to save fold models (e.g. models/lightgbm)",
    )
    ap.add_argument(
        "--n_splits",
        type=int,
        default=5,
        help="Number of CV folds (default: 5)",
    )
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    results_dir = "results/lightgbm"
    os.makedirs(results_dir, exist_ok=True)

    print(f"Loading metadata: {args.metadata}")
    df = pd.read_csv(args.metadata)

    if "aqi_continuous" not in df.columns:
        raise ValueError("metadata must contain 'aqi_continuous' as label.")

    # keep rows with valid label
    df = df[df["aqi_continuous"].notna()].reset_index(drop=True)
    print(f"Valid rows with aqi_continuous: {len(df)}")

    # build features
    X, feat_cols = get_features(df)
    y = df["aqi_continuous"].values.astype(float)

    print(f"Using {len(feat_cols)} features:")
    for c in feat_cols:
        print("  -", c)

    kf = KFold(n_splits=args.n_splits, shuffle=True, random_state=42)

    # conservative, light params for MacBook Air
    params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "min_data_in_leaf": 20,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "lambda_l1": 0.0,
        "lambda_l2": 0.0,
        "n_estimators": 300,
        "verbosity": -1,
        # LightGBM will auto-handle threads; dataset is small so this is safe
    }

    maes, rmses, r2s = [], [], []

    print(f"\n==== Training {args.n_splits}-Fold LightGBM AQI model ====\n")

    for fold, (train_idx, val_idx) in enumerate(kf.split(X), start=0):
        print(f"--- Fold {fold} ---")

        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        model = lgb.LGBMRegressor(**params)
        model.fit(X_tr, y_tr)

        pred = model.predict(X_val)

        mae = mean_absolute_error(y_val, pred)
        rmse = compute_rmse(y_val, pred)
        r2 = r2_score(y_val, pred)

        maes.append(mae)
        rmses.append(rmse)
        r2s.append(r2)

        print(f"MAE={mae:.3f} | RMSE={rmse:.3f} | R²={r2:.3f}")

        # save fold model
        fold_model_path = os.path.join(args.out_dir, f"weather_fold{fold}.txt")
        model.booster_.save_model(fold_model_path)
        print("Saved model:", fold_model_path)

        # save feature importance for this fold
        importance = model.booster_.feature_importance(importance_type="gain")
        fi_df = pd.DataFrame({"feature": feat_cols, "importance": importance})
        fi_df = fi_df.sort_values("importance", ascending=False)

        fi_path = os.path.join(results_dir, f"feature_importance_fold{fold}.csv")
        fi_df.to_csv(fi_path, index=False)
        print("Feature importance saved:", fi_path)
        print()

    # aggregate CV metrics
    mae_mean, mae_std = np.mean(maes), np.std(maes)
    rmse_mean, rmse_std = np.mean(rmses), np.std(rmses)
    r2_mean, r2_std = np.mean(r2s), np.std(r2s)

    print("==== CV Summary (LightGBM AQI) ====")
    print(f"MAE : {mae_mean:.3f} ± {mae_std:.3f}")
    print(f"RMSE: {rmse_mean:.3f} ± {rmse_std:.3f}")
    print(f"R²  : {r2_mean:.3f} ± {r2_std:.3f}")

    # save CV metrics
    cv_path = os.path.join(results_dir, "lightgbm_cv_metrics.txt")
    with open(cv_path, "w") as f:
        f.write(f"MAE_mean: {mae_mean:.4f}\n")
        f.write(f"MAE_std: {mae_std:.4f}\n")
        f.write(f"RMSE_mean: {rmse_mean:.4f}\n")
        f.write(f"RMSE_std: {rmse_std:.4f}\n")
        f.write(f"R2_mean: {r2_mean:.4f}\n")
        f.write(f"R2_std: {r2_std:.4f}\n")

    print("\nCV metrics saved:", cv_path)
    print("\n==== DONE 5-FOLD NUMERIC MODEL TRAINING ====")


if __name__ == "__main__":
    main()

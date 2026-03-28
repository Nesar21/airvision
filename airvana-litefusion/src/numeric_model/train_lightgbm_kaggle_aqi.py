#!/usr/bin/env python3
"""
train_lightgbm_kaggle_aqi.py

Train a LightGBM regression model to predict overall AQI
from pollutant-wise AQI indices (Kaggle Real-time AQI India 2023–2025).

Input:
    --metadata data/AQI_wide_with_aqi.csv
Output:
    Fold models: models/lightgbm_kaggle/aqi_fold{0..4}.txt
    CV metrics:  results/lightgbm_kaggle/kaggle_aqi_cv_metrics.txt
"""

import os
import argparse
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import lightgbm as lgb


def build_features(df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    candidate_cols = [
        "PM2.5",
        "PM10",
        "NO2",
        "SO2",
        "CO",
        "OZONE",
        "NH3",
    ]
    cols = [c for c in candidate_cols if c in df.columns]
    if not cols:
        raise ValueError("No pollutant feature columns found in metadata.")

    X = df[cols].astype(float)
    X = X.fillna(X.median(numeric_only=True))

    return X.values, cols


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mse = mean_squared_error(y_true, y_pred)
    return float(np.sqrt(mse))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", required=True,
                    help="Path to AQI_wide_with_aqi.csv")
    ap.add_argument("--out_dir", required=True,
                    help="Dir to save fold models (e.g. models/lightgbm_kaggle)")
    ap.add_argument("--n_splits", type=int, default=5)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    results_dir = "results/lightgbm_kaggle"
    os.makedirs(results_dir, exist_ok=True)

    print(f"Loading metadata: {args.metadata}")
    df = pd.read_csv(args.metadata)

    if "aqi_overall" not in df.columns:
        raise ValueError("metadata must contain 'aqi_overall' as label.")

    # label
    y = df["aqi_overall"].astype(float).values

    # features
    X, feat_cols = build_features(df)

    print(f"Using {len(feat_cols)} features:")
    for c in feat_cols:
        print("  -", c)

    kf = KFold(n_splits=args.n_splits, shuffle=True, random_state=42)

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
    }

    maes, rmses, r2s = [], [], []

    print(f"\n==== Training {args.n_splits}-Fold LightGBM AQI model (Kaggle) ====\n")

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X), start=0):
        print(f"--- Fold {fold} ---")
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        model = lgb.LGBMRegressor(**params)
        model.fit(X_tr, y_tr)

        pred = model.predict(X_va)

        mae = mean_absolute_error(y_va, pred)
        rmse = compute_rmse(y_va, pred)
        r2 = r2_score(y_va, pred)

        maes.append(mae)
        rmses.append(rmse)
        r2s.append(r2)

        print(f"Fold {fold}: MAE={mae:.3f} | RMSE={rmse:.3f} | R²={r2:.3f}")

        fold_path = os.path.join(args.out_dir, f"aqi_fold{fold}.txt")
        model.booster_.save_model(fold_path)
        print("Saved model:", fold_path)
        print()

    mae_mean, mae_std = np.mean(maes), np.std(maes)
    rmse_mean, rmse_std = np.mean(rmses), np.std(rmses)
    r2_mean, r2_std = np.mean(r2s), np.std(r2s)

    print("==== CV Summary (Kaggle AQI) ====")
    print(f"MAE : {mae_mean:.3f} ± {mae_std:.3f}")
    print(f"RMSE: {rmse_mean:.3f} ± {rmse_std:.3f}")
    print(f"R²  : {r2_mean:.3f} ± {r2_std:.3f}")

    cv_path = os.path.join(results_dir, "kaggle_aqi_cv_metrics.txt")
    with open(cv_path, "w") as f:
        f.write(f"MAE_mean: {mae_mean:.4f}\n")
        f.write(f"MAE_std: {mae_std:.4f}\n")
        f.write(f"RMSE_mean: {rmse_mean:.4f}\n")
        f.write(f"RMSE_std: {rmse_std:.4f}\n")
        f.write(f"R2_mean: {r2_mean:.4f}\n")
        f.write(f"R2_std: {r2_std:.4f}\n")

    print("\nCV metrics saved:", cv_path)
    print("\n==== DONE KAGGLE NUMERIC AQI TRAINING ====")


if __name__ == "__main__":
    main()

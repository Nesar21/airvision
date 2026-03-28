#!/usr/bin/env python3
import os
import json
import argparse
import numpy as np
import pandas as pd

import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", required=True,
                    help="CSV with AQI, *_norm, season_idx, day_idx, emb_*")
    ap.add_argument("--norm_stats", required=True,
                    help="Not used directly here, kept for interface consistency")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--learning_rate", type=float, default=0.03)
    ap.add_argument("--num_leaves", type=int, default=31)
    ap.add_argument("--n_estimators", type=int, default=2000)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ---------------------------------------------------------
    # Load metadata + checks
    # ---------------------------------------------------------
    df = pd.read_csv(args.metadata)

    num_cols = [
        "PM2.5_norm",
        "PM10_norm",
        "Temperature_norm",
        "Humidity_norm",
        "season_idx",
        "day_idx",
    ]
    for c in num_cols:
        if c not in df.columns:
            raise ValueError(f"Required numeric column '{c}' missing in dataset")

    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    if not emb_cols:
        raise ValueError("No embedding columns (emb_*) found in metadata")

    print(f"Found {len(emb_cols)} embedding columns")

    X = df[num_cols + emb_cols].values
    y = df["aqi"].values.astype("float32")

    kf = KFold(n_splits=args.folds, shuffle=True, random_state=42)

    all_rmse, all_mae = [], []
    boosters = []  # store booster per fold

    # ---------------------------------------------------------
    # K-Fold training
    # ---------------------------------------------------------
    for fold, (tr_idx, te_idx) in enumerate(kf.split(X), start=0):
        print(f"\n==== Fold {fold} ====")

        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        model = lgb.LGBMRegressor(
            learning_rate=args.learning_rate,
            num_leaves=args.num_leaves,
            n_estimators=args.n_estimators,
            objective="regression_l2",
        )

        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_te, y_te)],
            eval_metric="rmse",
        )

        preds = model.predict(X_te)

        rmse = np.sqrt(mean_squared_error(y_te, preds))
        mae = mean_absolute_error(y_te, preds)

        print(f"Fold {fold}: RMSE={rmse:.3f} | MAE={mae:.3f}")

        all_rmse.append(rmse)
        all_mae.append(mae)
        boosters.append(model.booster_)

    # ---------------------------------------------------------
    # Summary + save best booster
    # ---------------------------------------------------------
    mean_rmse = float(np.mean(all_rmse))
    std_rmse = float(np.std(all_rmse))
    mean_mae = float(np.mean(all_mae))
    std_mae = float(np.std(all_mae))

    print("\n==== Fusion Model Summary ====")
    print(f"RMSE: {mean_rmse:.3f} ± {std_rmse:.3f}")
    print(f"MAE : {mean_mae:.3f} ± {std_mae:.3f}")

    # Save metrics to text
    metrics_path = os.path.join(args.out_dir, "fusion_cv_metrics.txt")
    with open(metrics_path, "w") as f:
        f.write(f"RMSE: {mean_rmse:.4f} ± {std_rmse:.4f}\n")
        f.write(f"MAE : {mean_mae:.4f} ± {std_mae:.4f}\n")
        for i, (r, m) in enumerate(zip(all_rmse, all_mae)):
            f.write(f"Fold {i}: RMSE={r:.4f}, MAE={m:.4f}\n")
    print(f"Saved CV metrics → {metrics_path}")

    # Pick best fold booster by RMSE
    best_idx = int(np.argmin(all_rmse))
    best_booster = boosters[best_idx]

    best_model_path = os.path.join(args.out_dir, "best_fusion_model.txt")
    best_booster.save_model(best_model_path)
    print(f"Saved best LightGBM booster (fold {best_idx}) → {best_model_path}")


if __name__ == "__main__":
    main()

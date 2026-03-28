#!/usr/bin/env python3
"""
evaluate.py

PHASE 9A — System-level evaluation for AirVana LiteFusion.

This script evaluates:
  - image-only AQI (MobileNetV3 regression; uses mnv3_fold0.pt)
  - numeric-only AQI (LightGBM weather+satellite ensemble, folds 0–4)
  - news-only AQI (text_predictor)
  - full LiteFusion (image + numeric + news)

On:
  - Main dataset: data/metadata_sat_features.csv  (uses aqi_continuous)
  - Held-out image-only test set: data/metadata_test.csv (uses aqi)

It computes:
  - Overall MAE / RMSE / R²
  - Metrics by country (ignores countries with <50 samples)
  - Day vs Night (via sat_brightness)
  - Clear vs Hazy (via aqi_continuous threshold)

Usage:

    source venv/bin/activate
    export PYTHONPATH=$(pwd)

    python3 src/evaluation/evaluate.py \
        --aqi_meta data/metadata_sat_features.csv \
        --results_dir results/evaluation
"""

import os
import argparse
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd

import torch
from PIL import Image

import lightgbm as lgb

from src.fusion.litefusion_api import LiteFusionPredictor
from src.text_model.text_predictor import predict_from_news
from src.image_model.mobilenet_predictor import MobileNetAQIPredictor


# -----------------------------
# Metrics helpers
# -----------------------------

def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_mean = float(np.mean(y_true))
    ss_tot = float(np.sum((y_true - y_mean) ** 2))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    if ss_tot == 0.0:
        return 0.0
    return 1.0 - ss_res / ss_tot


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "MAE": compute_mae(y_true, y_pred),
        "RMSE": compute_rmse(y_true, y_pred),
        "R2": compute_r2(y_true, y_pred),
    }


# -----------------------------
# Model loading helpers
# -----------------------------

def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_image_model(device: torch.device) -> MobileNetAQIPredictor:
    """
    Load the trained MobileNetV3 regression model.

    We use mnv3_fold0.pt from your 5-fold training:
        models/mobilenet/mnv3_fold0.pt
    """
    model_path = "models/mobilenet/mnv3_fold0.pt"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Image model not found: {model_path}")

    # NOTE: MobileNetAQIPredictor signature: (model_path, device=None)
    predictor = MobileNetAQIPredictor(model_path=model_path, device=device)
    return predictor


def load_numeric_ensemble() -> Optional[list]:
    """Load LightGBM weather+sat ensemble (folds 0..4). Returns list of boosters or None."""
    models = []
    for i in range(5):
        path = f"models/lightgbm/weather_fold{i}.txt"
        if os.path.exists(path):
            models.append(lgb.Booster(model_file=path))
    if not models:
        return None
    return models


def numeric_features(df: pd.DataFrame) -> Tuple[np.ndarray, list]:
    cols = [
        "pm25_fetch", "pm10_fetch",
        "temp_fetch", "humidity_fetch", "pressure_fetch",
        "visibility_fetch", "wind_fetch", "wind_deg_fetch",
        "feels_like_fetch", "dew_point_fetch",
        "sat_brightness", "sat_blur", "sat_color_skew",
    ]
    cols = [c for c in cols if c in df.columns]
    if not cols:
        raise ValueError("No numeric feature columns found for LightGBM evaluation.")

    X = df[cols].astype(float)
    X = X.fillna(X.median(numeric_only=True))
    return X.values, cols


def predict_numeric(models: Optional[list], X: np.ndarray) -> np.ndarray:
    """Average predictions from LightGBM boosters. If models is None, return NaNs."""
    if models is None:
        return np.full(X.shape[0], np.nan, dtype=float)

    preds = []
    for booster in models:
        p = booster.predict(X)
        preds.append(p)
    return np.mean(np.vstack(preds), axis=0)


# -----------------------------
# Inference pipelines
# -----------------------------

def predict_images(df: pd.DataFrame, predictor: MobileNetAQIPredictor) -> np.ndarray:
    """
    Image-only inference, row-by-row.

    Uses predictor.predict(img_path) → (aqi, sigma_img).
    If any error occurs for a row, returns NaN for that sample.
    """
    paths = df["image_path"].tolist()
    preds = np.full(len(paths), np.nan, dtype=float)

    for i, path in enumerate(paths):
        try:
            aqi, _sigma = predictor.predict(path)
            preds[i] = float(aqi)
        except Exception:
            preds[i] = np.nan

    return preds


def predict_news(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Use text_predictor on 'news' column. Embeddings not used (None)."""
    aqi_txt_list = []
    conf_list = []

    for _, row in df.iterrows():
        # Force to string, handle NaN safely
        headlines = row.get("news", "")
        if not isinstance(headlines, str):
            headlines = str(headlines) if pd.notna(headlines) else ""

        aqi_txt, conf_txt, _tags = predict_from_news(headlines, None)
        aqi_txt_list.append(float(aqi_txt))
        conf_list.append(float(conf_txt))

    return np.array(aqi_txt_list, dtype=float), np.array(conf_list, dtype=float)



# -----------------------------
# Fusion + grouped evaluation
# -----------------------------

def fuse_all(
    df: pd.DataFrame,
    y_true: np.ndarray,
    aqi_img: np.ndarray,
    aqi_num: np.ndarray,
    aqi_txt: np.ndarray,
    conf_txt: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    """
    Returns metrics for:
      - image_only
      - numeric_only
      - news_only
      - fusion_full
    """
    fusion = LiteFusionPredictor()

    # Image-only
    mask_img = np.isfinite(aqi_img)
    y_img = aqi_img[mask_img]
    t_img = y_true[mask_img]
    metrics_img = compute_metrics(t_img, y_img)

    # Numeric-only (may be NaN if no LightGBM models)
    mask_num = np.isfinite(aqi_num)
    if mask_num.any():
        y_num = aqi_num[mask_num]
        t_num = y_true[mask_num]
        metrics_num = compute_metrics(t_num, y_num)
    else:
        metrics_num = {"MAE": np.nan, "RMSE": np.nan, "R2": np.nan}

    # News-only (just aqi_txt)
    mask_txt = np.isfinite(aqi_txt)
    y_news = aqi_txt[mask_txt]
    t_news = y_true[mask_txt]
    metrics_news = compute_metrics(t_news, y_news)

    # Full LiteFusion (image + numeric + news, where available)
    fused_preds = []
    fused_truth = []

    for i in range(len(df)):
        mods = {}

        if np.isfinite(aqi_img[i]):
            mods["image"] = (float(aqi_img[i]), 0.0)

        if np.isfinite(aqi_num[i]):
            mods["numeric"] = (float(aqi_num[i]), 0.0)

        if np.isfinite(aqi_txt[i]):
            mods["news"] = (float(aqi_txt[i]), float(conf_txt[i]))

        fused = fusion.fuse_aqi(
            image=mods.get("image"),
            numeric=mods.get("numeric"),
            news=mods.get("news"),
        )

        if fused is None:
            continue

        fused_preds.append(float(fused.mean))
        fused_truth.append(float(y_true[i]))

    fused_preds = np.array(fused_preds, dtype=float)
    fused_truth = np.array(fused_truth, dtype=float)
    metrics_fused = compute_metrics(fused_truth, fused_preds)

    return {
        "image_only": metrics_img,
        "numeric_only": metrics_num,
        "news_only": metrics_news,
        "fusion_full": metrics_fused,
    }


def grouped_metrics(
    df: pd.DataFrame,
    y_true: np.ndarray,
    aqi_img: np.ndarray,
    aqi_num: np.ndarray,
    aqi_txt: np.ndarray,
    conf_txt: np.ndarray,
    group_name: str,
    group_values: np.ndarray,
    min_count: int = 50,
) -> pd.DataFrame:
    """
    Compute metrics per group (country / day-night / clear-hazy).
    group_values: 1D array with group labels for each row.
    """
    fusion = LiteFusionPredictor()
    results = []

    unique_groups = pd.unique(group_values)
    for g in unique_groups:
        mask = (group_values == g)
        if mask.sum() < min_count:
            continue

        y_g = y_true[mask]
        img_g = aqi_img[mask]
        num_g = aqi_num[mask]
        txt_g = aqi_txt[mask]
        conf_g = conf_txt[mask]

        # image-only
        m_img = np.isfinite(img_g)
        if m_img.any():
            metrics_img = compute_metrics(y_g[m_img], img_g[m_img])
        else:
            metrics_img = {"MAE": np.nan, "RMSE": np.nan, "R2": np.nan}

        # numeric-only
        m_num = np.isfinite(num_g)
        if m_num.any():
            metrics_num = compute_metrics(y_g[m_num], num_g[m_num])
        else:
            metrics_num = {"MAE": np.nan, "RMSE": np.nan, "R2": np.nan}

        # news-only
        m_txt = np.isfinite(txt_g)
        if m_txt.any():
            metrics_news = compute_metrics(y_g[m_txt], txt_g[m_txt])
        else:
            metrics_news = {"MAE": np.nan, "RMSE": np.nan, "R2": np.nan}

        # fusion
        fused_preds = []
        fused_truth = []
        for i in range(len(y_g)):
            mods = {}
            if np.isfinite(img_g[i]):
                mods["image"] = (float(img_g[i]), 0.0)
            if np.isfinite(num_g[i]):
                mods["numeric"] = (float(num_g[i]), 0.0)
            if np.isfinite(txt_g[i]):
                mods["news"] = (float(txt_g[i]), float(conf_g[i]))

            fused = fusion.fuse_aqi(
                image=mods.get("image"),
                numeric=mods.get("numeric"),
                news=mods.get("news"),
            )
            if fused is None:
                continue
            fused_preds.append(float(fused.mean))
            fused_truth.append(float(y_g[i]))

        if len(fused_preds) > 0:
            metrics_fused = compute_metrics(
                np.array(fused_truth, dtype=float),
                np.array(fused_preds, dtype=float),
            )
        else:
            metrics_fused = {"MAE": np.nan, "RMSE": np.nan, "R2": np.nan}

        results.append({
            group_name: g,
            "count": int(mask.sum()),

            "img_MAE": metrics_img["MAE"],
            "img_RMSE": metrics_img["RMSE"],
            "img_R2": metrics_img["R2"],

            "num_MAE": metrics_num["MAE"],
            "num_RMSE": metrics_num["RMSE"],
            "num_R2": metrics_num["R2"],

            "news_MAE": metrics_news["MAE"],
            "news_RMSE": metrics_news["RMSE"],
            "news_R2": metrics_news["R2"],

            "fusion_MAE": metrics_fused["MAE"],
            "fusion_RMSE": metrics_fused["RMSE"],
            "fusion_R2": metrics_fused["R2"],
        })

    return pd.DataFrame(results)


# -----------------------------
# Held-out test-set evaluation
# -----------------------------

def evaluate_external_test(
    img_model: MobileNetAQIPredictor,
    device: torch.device,
    results_dir: str,
):
    """
    Evaluate image-only performance on held-out test images:
        data/metadata_test.csv

    Uses:
        - image_path
        - aqi        (scalar, e.g. 50 in SAPID rows)

    Numeric and fusion evaluation are skipped because test metadata
    does not contain *_fetch or sat_* features.
    """
    test_meta_path = "data/metadata_test.csv"
    if not os.path.exists(test_meta_path):
        print("\nNo data/metadata_test.csv found → skipping held-out test evaluation.")
        return

    print(f"\nLoading held-out test metadata from {test_meta_path}")
    df_test = pd.read_csv(test_meta_path)

    if "aqi" not in df_test.columns:
        print("No 'aqi' column in metadata_test.csv → skipping test evaluation.")
        return

    df_test = df_test[df_test["aqi"].notna()].reset_index(drop=True)
    if len(df_test) == 0:
        print("No valid 'aqi' labels in metadata_test.csv → skipping test evaluation.")
        return

    print("Held-out test samples:", len(df_test))

    # Ground truth: use 'aqi' column
    y_test = df_test["aqi"].astype(float).values

    print("Predicting AQI from images on held-out test set...")
    aqi_img_test = predict_images(df_test, img_model)

    mask = np.isfinite(aqi_img_test)
    if not mask.any():
        print("All held-out image predictions are NaN → cannot compute test metrics.")
        return

    metrics_test_img = compute_metrics(y_test[mask], aqi_img_test[mask])

    test_out_path = os.path.join(results_dir, "testset_image_only_metrics.txt")
    with open(test_out_path, "w") as f:
        f.write("[image_only_test]\n")
        f.write(f"MAE : {metrics_test_img['MAE']:.3f}\n")
        f.write(f"RMSE: {metrics_test_img['RMSE']:.3f}\n")
        f.write(f"R2  : {metrics_test_img['R2']:.3f}\n")

    print("Held-out test image-only metrics saved →", test_out_path)


# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aqi_meta", required=True,
                    help="Metadata CSV with aqi_continuous, sat_* and *_fetch columns "
                         "(e.g. data/metadata_sat_features.csv)")
    ap.add_argument("--results_dir", required=True,
                    help="Directory to save evaluation outputs (e.g. results/evaluation)")
    args = ap.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    # 1) Load main metadata
    print(f"Loading AQI metadata from {args.aqi_meta}")
    df = pd.read_csv(args.aqi_meta)

    if "aqi_continuous" not in df.columns:
        raise ValueError("metadata must contain 'aqi_continuous' column.")

    df = df[df["aqi_continuous"].notna()].reset_index(drop=True)
    print("Rows with valid AQI labels:", len(df))

    y_true = df["aqi_continuous"].values.astype(float)

    # 2) Load models
    device = get_device()
    print("Using device:", device)

    print("Loading image model (MobileNetV3 regression, mnv3_fold0.pt)...")
    img_model = load_image_model(device)

    print("Loading LightGBM numeric ensemble (if available)...")
    num_models = load_numeric_ensemble()
    if num_models is None:
        print("  No LightGBM models found → numeric-only and numeric part of fusion will be disabled.")
    else:
        print(f"  Loaded {len(num_models)} LightGBM boosters.")

    # 3) Predict image AQI on main metadata
    print("\nPredicting AQI from images (image-only, main dataset)...")
    aqi_img = predict_images(df, img_model)

    # 4) Predict numeric AQI (weather+sat) on main metadata
    print("Predicting AQI from numeric (weather+sat) model (main dataset)...")
    X_num, feat_cols = numeric_features(df)
    aqi_num = predict_numeric(num_models, X_num)

    # 5) Predict news tone AQI on main metadata
    print("Predicting AQI from news tone module (main dataset)...")
    aqi_txt, conf_txt = predict_news(df)

    # 6) Overall metrics on main dataset
    print("\nComputing overall metrics (main dataset)...")
    metrics_all = fuse_all(df, y_true, aqi_img, aqi_num, aqi_txt, conf_txt)

    overall_path = os.path.join(args.results_dir, "overall_metrics.txt")
    with open(overall_path, "w") as f:
        for model_name, m in metrics_all.items():
            f.write(f"[{model_name}]\n")
            f.write(f"MAE : {m['MAE']:.3f}\n")
            f.write(f"RMSE: {m['RMSE']:.3f}\n")
            f.write(f"R2  : {m['R2']:.3f}\n\n")

    print("Overall metrics (main dataset) saved →", overall_path)

    # 7) Country-level metrics (Global generalization view)
    print("\nComputing metrics by country (main dataset)...")
    if "country" not in df.columns:
        print("No 'country' column found; skipping country-level analysis.")
    else:
        countries = df["country"].fillna("Unknown").astype(str).values
        df_country = grouped_metrics(
            df, y_true, aqi_img, aqi_num, aqi_txt, conf_txt,
            group_name="country", group_values=countries, min_count=50,
        )
        country_path = os.path.join(args.results_dir, "metrics_by_country.csv")
        df_country.to_csv(country_path, index=False)
        print("Country-level metrics saved →", country_path)
        print("  (Look at 'China' vs others if present.)")

    # 8) Day vs Night metrics (via sat_brightness)
    print("\nComputing Day vs Night metrics (via sat_brightness, main dataset)...")
    if "sat_brightness" in df.columns:
        brightness = df["sat_brightness"].astype(float)
        brightness = brightness.fillna(brightness.median())
        thr_b = float(brightness.median())
        labels_day_night = np.where(brightness.values >= thr_b, "day_like", "night_like")

        df_dn = grouped_metrics(
            df, y_true, aqi_img, aqi_num, aqi_txt, conf_txt,
            group_name="time_segment", group_values=labels_day_night, min_count=50,
        )
        dn_path = os.path.join(args.results_dir, "metrics_day_night.csv")
        df_dn.to_csv(dn_path, index=False)
        print("Day/Night metrics saved →", dn_path)
    else:
        print("No 'sat_brightness' column; skipping day/night analysis.")

    # 9) Clear vs Hazy metrics (via aqi_continuous threshold)
    print("\nComputing Clear vs Hazy metrics (via aqi_continuous threshold, main dataset)...")
    thr_aqi = 75.0  # <= 75 → clear, >75 → hazy
    labels_ch = np.where(y_true <= thr_aqi, "clear_like", "hazy_like")

    df_ch = grouped_metrics(
        df, y_true, aqi_img, aqi_num, aqi_txt, conf_txt,
        group_name="condition", group_values=labels_ch, min_count=50,
    )
    ch_path = os.path.join(args.results_dir, "metrics_clear_hazy.csv")
    df_ch.to_csv(ch_path, index=False)
    print("Clear/Hazy metrics saved →", ch_path)

    # 10) Held-out test-set evaluation (image-only)
    evaluate_external_test(img_model, device, args.results_dir)

    print("\nPHASE 9A evaluation complete.")


if __name__ == "__main__":
    main()

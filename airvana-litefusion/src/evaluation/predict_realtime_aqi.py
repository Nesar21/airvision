#!/usr/bin/env python3
"""
predict_realtime_aqi.py

Real-time AQI prediction using:
  - Image model (MobileNetV3 AQI regressor, 5-fold ensemble)
  - Numeric model (Kaggle-trained LightGBM on pollutants → AQI)
  - LiteFusion uncertainty-based fusion
  - WAQI API for live pollutants + official AQI

Usage example:

    export PYTHONPATH=$(pwd)
    export WAQI_TOKEN="YOUR_WAQI_TOKEN"

    python3 src/evaluation/predict_realtime_aqi.py \
        --img data/images/manual_test/Delhi-Air-Pollution-1-1.jpg \
        --lat 28.6129 \
        --lon 77.2295

Required:
  - models/mobilenet/mnv3_fold0.pt ... mnv3_fold4.pt
  - models/lightgbm_kaggle/aqi_fold0.txt ... aqi_fold4.txt
  - WAQI_TOKEN env var or --waqi_token argument
"""

import os
import argparse
import requests
import numpy as np

from src.image_model.mobilenet_predictor import MobileNetAQIPredictor
from src.numeric_model.kaggle_numeric_predictor import KaggleNumericAQIPredictor
from src.fusion.litefusion_api import LiteFusionPredictor


def fetch_waqi(lat: float, lon: float, token: str):
    """
    Call WAQI API for a given lat/lon and return:
      - official AQI
      - pollutant dict: {pm25, pm10, no2, so2, co, o3, nh3}
    """
    url = f"https://api.waqi.info/feed/geo:{lat};{lon}/?token={token}"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    if data.get("status") != "ok":
        raise RuntimeError(f"WAQI error: {data.get('data')}")

    d = data["data"]
    aqi_official = d.get("aqi", None)

    iaqi = d.get("iaqi", {}) or {}

    def get_iaqi(key):
        val = iaqi.get(key)
        if isinstance(val, dict):
            return val.get("v")
        return None

    pollutants = {
        "pm25": get_iaqi("pm25"),
        "pm10": get_iaqi("pm10"),
        "no2":  get_iaqi("no2"),
        "so2":  get_iaqi("so2"),
        "co":   get_iaqi("co"),
        "o3":   get_iaqi("o3"),
        "nh3":  get_iaqi("nh3"),  # may be missing often
    }

    return aqi_official, pollutants, d


def predict_image_aqi(img_path: str, device: str = "mps") -> (float, list):
    """
    Run 5-fold MobileNet ensemble on a single image_path and return:
      - ensemble AQI
      - list of per-model predictions
    """
    model_paths = [
        "models/mobilenet/mnv3_fold0.pt",
        "models/mobilenet/mnv3_fold1.pt",
        "models/mobilenet/mnv3_fold2.pt",
        "models/mobilenet/mnv3_fold3.pt",
        "models/mobilenet/mnv3_fold4.pt",
    ]

    preds = []
    for mp in model_paths:
        if not os.path.exists(mp):
            raise FileNotFoundError(f"Missing image model checkpoint: {mp}")
        model = MobileNetAQIPredictor(mp, device=device)
        p, _ = model.predict(img_path)
        preds.append(p)

    return float(np.mean(preds)), preds


def predict_numeric_aqi_from_pollutants(pollutants: dict) -> float:
    """
    Use KaggleNumericAQIPredictor to estimate AQI from pollutants dict.
    Pollutant keys expected: pm25, pm10, no2, so2, co, o3, nh3
    Missing values are allowed (treated as 0.0).
    """
    num_pred = KaggleNumericAQIPredictor()

    aqi_num = num_pred.predict_aqi(
        pm25=pollutants.get("pm25"),
        pm10=pollutants.get("pm10"),
        no2=pollutants.get("no2"),
        so2=pollutants.get("so2"),
        co=pollutants.get("co"),
        o3=pollutants.get("o3"),
        nh3=pollutants.get("nh3"),
    )
    return aqi_num


def main():
    ap = argparse.ArgumentParser(description="Real-time AQI prediction via image + WAQI + LiteFusion")
    ap.add_argument("--img", required=True, help="Path to input image")
    ap.add_argument("--lat", type=float, required=True, help="Latitude of location")
    ap.add_argument("--lon", type=float, required=True, help="Longitude of location")
    ap.add_argument("--waqi_token", type=str, default=None, help="WAQI API token (or set WAQI_TOKEN env)")
    ap.add_argument("--device", type=str, default="mps", help="Torch device: mps or cpu")

    args = ap.parse_args()

    if not os.path.exists(args.img):
        raise FileNotFoundError(f"Input image not found: {args.img}")

    token = args.waqi_token or os.getenv("WAQI_TOKEN")
    if not token:
        raise RuntimeError("Missing WAQI token. Use --waqi_token or set WAQI_TOKEN env var.")

    print("=== Step 1: Image-only prediction (MobileNetV3 ensemble) ===")
    aqi_img, img_preds = predict_image_aqi(args.img, device=args.device)
    print(f"Per-fold image AQI: {img_preds}")
    print(f"Image-only ensemble AQI: {aqi_img:.2f}")

    print("\n=== Step 2: Fetch pollutants from WAQI ===")
    aqi_official, pollutants, raw = fetch_waqi(args.lat, args.lon, token)
    print(f"WAQI official AQI: {aqi_official}")
    print("Pollutants (IAQI or index-like):")
    for k, v in pollutants.items():
        print(f"  {k.upper():4s}: {v}")

    print("\n=== Step 3: Numeric AQI from Kaggle model ===")
    aqi_num = predict_numeric_aqi_from_pollutants(pollutants)
    print(f"Numeric-only AQI (Kaggle model): {aqi_num:.2f}")

    print("\n=== Step 4: LiteFusion (high numeric confidence) ===")
    fusion = LiteFusionPredictor(
        base_sigma_image=12.0,   # image less confident (esp. at night)
        base_sigma_numeric=6.0,  # numeric more confident
        base_sigma_news=30.0,
        base_sigma_satellite=20.0,
        base_sigma_pm25aqi=10.0,
    )

    fused = fusion.fuse_aqi(
        image=(aqi_img, None),    # None → fall back to base_sigma_image
        numeric=(aqi_num, None),  # None → fall back to base_sigma_numeric
    )

    print("\n=== Step 5: Final comparison ===")
    print(f"Image-only AQI:        {aqi_img:.2f}")
    print(f"Numeric-only AQI:      {aqi_num:.2f}")
    print(f"LiteFusion AQI:        {fused.mean:.2f}")
    print(f"LiteFusion sigma:      {fused.sigma:.2f}")
    print(f"LiteFusion weights:    {fused.norm_weights}")
    print(f"WAQI official AQI:     {aqi_official}")

    print("\nDone.")


if __name__ == "__main__":
    main()

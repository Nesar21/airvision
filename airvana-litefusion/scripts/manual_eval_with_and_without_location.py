#!/usr/bin/env python3
import os
import sys
import pandas as pd

# --------------------------------------
# PYTHONPATH fix
# --------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
for p in [PROJECT_ROOT, SRC_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

# --------------------------------------
# Use correct regression model loader
# --------------------------------------
from image_model.mobilenetv3_traqid_predictor import MobileNetTraqidPredictor


def main():
    print("\n=== Image-Only AQI Test: Regression Model ===\n")

    model = MobileNetTraqidPredictor(
        model_path="models/mobilenet/mnv3_regression.pt"
    )

    tests = [
        {"name": "Delhi manual", "path": "data/images/manual_test/Delhi-Air-Pollution-1-1.jpg"},
        {"name": "Mysore-1",     "path": "data/images/manual_test/Mysore.png"},
        {"name": "Mysore-2",     "path": "data/images/manual_test/Mysore-2.jpeg"},
        {"name": "Japan-night",  "path": "data/images/manual_test/Japan-night.png"},
    ]

    results = []
    for item in tests:
        name = item["name"]
        img = item["path"]

        print(f"→ Predicting: {name}")
        try:
            pred = float(model.predict(img))
        except Exception as e:
            print("  ERROR →", e)
            pred = None

        results.append({
            "name": name,
            "image_path": img,
            "aqi_pred_image_only": pred,
        })

        print(f"  Predicted AQI: {pred}\n")

    os.makedirs("results", exist_ok=True)
    out_path = "results/manual_test_image_aqi.csv"
    pd.DataFrame(results).to_csv(out_path, index=False)

    print("Saved →", out_path)
    print("\n✓ Complete\n")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import os
import sys

# Ensure project root + src/ on PYTHONPATH
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")

for p in [PROJECT_ROOT, SRC_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from fusion.traqid_fusion_predictor import TraqidFusionAQIPredictor


def main():
    print("Loading TRAQID Night Fusion Predictor...")

    model = TraqidFusionAQIPredictor(
        embeddings_csv="data/traqid_embeddings.csv",
        booster_path="models/traqid_night_aqi/best_fusion_model.txt"
    )

    # Sample TRAQID Image IDs from Sequence 2
    test_ids = [195, 200, 210, 215, 220]
    print(f"\nTesting {len(test_ids)} images:\n")

    for img_id in test_ids:
        try:
            pred, true = model.predict_image_id(img_id)
            print(f"Image {img_id}: Pred AQI = {pred:.2f} | True = {true:.2f}")
        except Exception as e:
            print(f"Image {img_id}: ERROR → {e}")

    print("\n✓ TRAQID Night Fusion test complete.\n")


if __name__ == "__main__":
    main()

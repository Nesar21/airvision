#!/usr/bin/env python3
import os
import torch
from PIL import Image
from torchvision import transforms
from src.image_model.haze_predictor import MobileNetHazePredictor

device = "mps" if torch.backends.mps.is_available() else "cpu"

model_paths = [
    "models/mobilenet_haze/mnv3_haze_fold0.pt",
    "models/mobilenet_haze/mnv3_haze_fold1.pt",
    "models/mobilenet_haze/mnv3_haze_fold2.pt",
    "models/mobilenet_haze/mnv3_haze_fold3.pt",
    "models/mobilenet_haze/mnv3_haze_fold4.pt",
]

predictor = MobileNetHazePredictor(model_paths, device=device)

test_images = [
    "data/images/manual_test/Delhi-Air-Pollution-1-1.jpg",
    "data/images/manual_test/Japan-night.png",
    "data/images/manual_test/Mysore.png",
    "data/images/manual_test/Mysore-2.jpeg",
]

print("\n=== Manual Haze Predictions ===")
for img in test_images:
    label, conf = predictor.predict(img)
    txt = ["CLEAR", "HAZY/SMOG"][label]
    print(f"{os.path.basename(img)} → {txt} (conf={conf:.3f})")

print("\n✓ Complete.")

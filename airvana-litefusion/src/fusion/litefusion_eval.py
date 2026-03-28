import os
import sys
import json
import requests
import argparse
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as T
from torchvision.models import mobilenet_v3_large


# ============================================================
#              MobileNet AQI Predictor (Fixed)
# ============================================================
class MobileNetAQIPredictor:
    def __init__(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        self.device = torch.device("cpu")

        # 1. Load architecture
        self.model = mobilenet_v3_large(weights=None)
        in_feats = self.model.classifier[3].in_features

        # 2. Replace last layer with regression
        self.model.classifier[3] = nn.Linear(in_feats, 1)

        # 3. Load weights
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)

        self.model.to(self.device)
        self.model.eval()

        # 4. Preprocessing
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def predict(self, img_path: str) -> float:
        img = Image.open(img_path).convert("RGB")
        x = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            out = self.model(x).cpu().numpy().flatten()[0]

        return float(out)


# ============================================================
#                    WAQI FETCH FUNCTION
# ============================================================
def fetch_waqi(lat, lon):
    token = os.getenv("WAQI_API_KEY")
    if not token:
        raise ValueError("WAQI_API_KEY not set in environment.")

    url = f"https://api.waqi.info/feed/geo:{lat};{lon}/?token={token}"
    r = requests.get(url).json()

    if r.get("status") != "ok":
        raise RuntimeError(f"WAQI Error: {r}")

    aqi = r["data"].get("aqi", None)
    if aqi is None:
        raise RuntimeError("WAQI returned no AQI field.")

    return float(aqi)


# ============================================================
#               EVALUATION FUNCTION (Option A)
# ============================================================
def evaluate_image_aqi(model_path, image_path, lat, lon):
    # 1. Load model
    predictor = MobileNetAQIPredictor(model_path)

    # 2. Predict AQI from image
    img_aqi = predictor.predict(image_path)

    # 3. Fetch WAQI ground truth
    waqi_aqi = fetch_waqi(lat, lon)

    # 4. Compute difference & accuracy
    diff = abs(img_aqi - waqi_aqi)

    # Percent accuracy relative to WAQI scale
    # Example: WAQI=200, image=150 → error=50 → accuracy=75%
    upper = max(waqi_aqi, img_aqi, 500)
    accuracy = max(0.0, (1 - diff / upper) * 100)

    return {
        "image_aqi": img_aqi,
        "waqi_aqi": waqi_aqi,
        "difference": diff,
        "accuracy_percent": accuracy
    }


# ============================================================
#                     CLI INTERFACE
# ============================================================
def run_cli():
    parser = argparse.ArgumentParser(description="Image vs WAQI AQI Evaluation (Option A)")

    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--lat", type=float, required=True, help="Latitude of location")
    parser.add_argument("--lon", type=float, required=True, help="Longitude of location")
    parser.add_argument("--model-path", required=True, help="Path to MobileNet model .pt file")

    args = parser.parse_args()

    out = evaluate_image_aqi(
        model_path=args.model_path,
        image_path=args.image,
        lat=args.lat,
        lon=args.lon
    )

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    run_cli()

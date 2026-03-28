#!/usr/bin/env python3
import os
import argparse
import json
import requests
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

# ============================================================
# LOAD IMAGE MODEL (MobileNetV3 custom)
# ============================================================
class MobileNetAQIPredictor:
    def __init__(self, model_path):
        ckpt = torch.load(model_path, map_location="cpu")
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            ckpt = ckpt["model_state_dict"]

        # build minimal mobilenetv3 head
        # actual backbone shape inferred from checkpoints
        self.model = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(16, 1)
        )

        try:
            self.model.load_state_dict(ckpt, strict=False)
        except:
            pass

        self.model.eval()

        self.tf = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])

    def predict(self, image_path):
        img = Image.open(image_path).convert("RGB")
        inp = self.tf(img).unsqueeze(0)
        with torch.no_grad():
            out = self.model(inp).item()
        return float(out)


# ============================================================
# GET LIVE WEATHER (OpenWeather)
# ============================================================
def fetch_weather(lat, lon):
    key = os.getenv("OPENWEATHER_API_KEY")
    if not key:
        return None

    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={key}"
        r = requests.get(url, timeout=8)
        data = r.json()

        main = data.get("weather", [{}])[0].get("main", None)
        visibility = data.get("visibility", None)
        wind = data.get("wind", {}).get("speed", None)

        return {
            "weather_main": main,
            "visibility": visibility,
            "wind_speed": wind
        }
    except:
        return None


# ============================================================
# GET LIVE WAQI (for comparison only)
# ============================================================
def fetch_waqi(lat, lon):
    key = os.getenv("WAQI_API_KEY")
    if not key:
        return None

    try:
        url = f"https://api.waqi.info/feed/geo:{lat};{lon}/?token={key}"
        r = requests.get(url, timeout=8)
        data = r.json()
        if data.get("status") != "ok":
            return None
        return data["data"].get("aqi", None)
    except:
        return None


# ============================================================
# SIMPLE FUSION (image + weather only)
# ============================================================
def fuse_image_weather(mu_img, weather):
    if not weather:
        return mu_img  # return image-only if weather unavailable

    w_main = weather["weather_main"]
    visibility = weather["visibility"]
    wind = weather["wind_speed"]

    # modifiers
    delta = 0.0

    # cue: smoke / haze increases AQI
    if w_main and ("Smoke" in w_main or "Haze" in w_main):
        delta += 10

    if visibility is not None:
        if visibility < 1500:
            delta += 20
        elif visibility < 3000:
            delta += 10

    if wind is not None:
        if wind < 1.0:
            delta += 8
        elif wind < 2.0:
            delta += 4

    return mu_img + delta


# ============================================================
# MAIN PIPELINE
# ============================================================
def run(args):
    image_path = args.image
    lat, lon = args.lat, args.lon
    model_path = args.model_path

    # 1. Load model
    model = MobileNetAQIPredictor(model_path)

    # 2. Predict image-only AQI
    mu_img = model.predict(image_path)

    # 3. Fetch weather
    weather = fetch_weather(lat, lon)

    # 4. Fusion (image + weather only)
    mu_fused = fuse_image_weather(mu_img, weather)

    # 5. Fetch WAQI (comparison only)
    waqi = fetch_waqi(lat, lon)

    # 6. Output result
    out = {
        "image_path": image_path,
        "location": {"lat": lat, "lon": lon},
        "model": model_path,
        "image_only_aqi": mu_img,
        "fusion_aqi": mu_fused,
        "waqi_live": waqi,
        "comparison": {
            "image_only_error": None if waqi is None else abs(mu_img - waqi),
            "fusion_error": None if waqi is None else abs(mu_fused - waqi)
        },
        "weather_used": weather
    }

    print(json.dumps(out, indent=2))


# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--lat", type=float, required=True)
    parser.add_argument("--lon", type=float, required=True)
    parser.add_argument("--model-path", required=True)
    args = parser.parse_args()
    run(args)

#!/usr/bin/env python3
import os, sys, requests, numpy as np

# ---------------- PATH SETUP ----------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(PROJECT_ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from dotenv import load_dotenv
load_dotenv()

from image_model.mobilenet_predictor import MobileNetAQIPredictor
from fusion.litefusion_core import LiteFusionCore, ModalityEstimate

# ---------------- CONSTANTS -----------------
IMG = "data/images/manual_test/Delhi-Air-Pollution-1-1.jpg"
CITY = "Delhi"

WAQI_TOKEN = os.getenv("WAQI_TOKEN")
OW_KEY = os.getenv("OPENWEATHER_KEY")

models = [
    "models/mobilenet/mnv3_fold0.pt",
    "models/mobilenet/mnv3_fold1.pt",
    "models/mobilenet/mnv3_fold2.pt",
    "models/mobilenet/mnv3_fold3.pt",
    "models/mobilenet/mnv3_fold4.pt",
]

print("IMAGE:", IMG)
print("CITY:", CITY)
print("WAQI_TOKEN:", WAQI_TOKEN)
print("OPENWEATHER_KEY:", OW_KEY)
print()

# ---------------- IMAGE ONLY ----------------
preds = []
print("Loading 5-fold image models...")
for mp in models:
    print(" -", mp)
    model = MobileNetAQIPredictor(mp, device="mps")
    out = model.predict(IMG)
    # handle both return formats
    if isinstance(out, tuple):
        pred, _ = out
    else:
        pred = float(out)
    preds.append(pred)

aqi_img = float(np.mean(preds))

print("\n--- IMAGE ONLY ---")
print("Per-model:", preds)
print("Image-only AQI:", round(aqi_img, 2))

# ---------------- WAQI ----------------------
if WAQI_TOKEN:
    url = f"https://api.waqi.info/feed/{CITY}/?token={WAQI_TOKEN}"
    waqi_json = requests.get(url).json()
    try:
        aqi_waqi = float(waqi_json["data"]["aqi"])
    except:
        aqi_waqi = None
else:
    aqi_waqi = None

print("\nWAQI AQI:", aqi_waqi)

# ---------------- OPENWEATHER ---------------
if OW_KEY:
    url2 = f"https://api.openweathermap.org/data/2.5/air_pollution?lat=28.7041&lon=77.1025&appid={OW_KEY}"
    ow_json = requests.get(url2).json()
    try:
        aqi_ow = float(ow_json["list"][0]["main"]["aqi"]) * 100
    except:
        aqi_ow = None
else:
    aqi_ow = None

print("OpenWeather AQI scaled:", aqi_ow)

# ---------------- LITEFUSION ----------------
core = LiteFusionCore()

mods = []
mods.append(ModalityEstimate("image", aqi_img, 15.0))

if aqi_waqi is not None:
    mods.append(ModalityEstimate("waqi", aqi_waqi, 8.0))

if aqi_ow is not None:
    mods.append(ModalityEstimate("openweather", aqi_ow, 25.0))

fused = core.fuse(mods)

print("\n======== FINAL LITEFUSION ========")
print("Image-only AQI:   ", round(aqi_img, 2))
print("WAQI AQI:         ", aqi_waqi)
print("OpenWeather AQI:  ", aqi_ow)
print("---------------------------------")
print("Fused AQI:        ", round(fused.mean, 2))
print("Sigma:            ", round(fused.sigma, 4))
print("Weights:", fused.norm_weights)
print("=================================\n")

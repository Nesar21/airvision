#!/usr/bin/env python3

import os, sys, requests, numpy as np
from dotenv import load_dotenv

# ------------------ PATH SETUP ------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(PROJECT_ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from image_model.mobilenet_predictor import MobileNetAQIPredictor
from image_model.haze_predictor import MobileNetHazePredictor
from text_model.text_predictor import predict_from_news
from fusion.litefusion_core import LiteFusionCore, ModalityEstimate

# ------------------ LOAD ENV --------------------
load_dotenv()
WAQI_TOKEN = os.getenv("WAQI_TOKEN")
OPENWEATHER_KEY = os.getenv("OPENWEATHER_KEY")

print("WAQI_TOKEN:", WAQI_TOKEN)
print("OPENWEATHER_KEY:", OPENWEATHER_KEY)


# ============================================================
# 1. IMAGE AQI (MobileNetV3)
# ============================================================
img_path = "data/images/manual_test/Delhi-Air-Pollution-1-1.jpg"
ckpt_img = "models/mobilenet/mnv3_fold0.pt"

model_img = MobileNetAQIPredictor(model_path=ckpt_img)
img_aqi, _ = model_img.predict(img_path)

# real evaluated sigma for image model
img_sigma = 8.0

print("\nIMAGE AQI:", img_aqi, "sigma:", img_sigma)


# ============================================================
# 2. HAZE MODEL (optional – does not affect AQI directly)
# ============================================================
try:
    haze_model = MobileNetHazePredictor(
        model_paths="models/mobilenet_haze_multisource/mnv3_haze_multisource_fold0.pt"
    )
    haze_label, haze_conf = haze_model.predict(img_path)
    haze_status = "CLEAR" if haze_label == 0 else "HAZY"
    print("HAZE:", haze_status, "confidence:", haze_conf)
except Exception as e:
    print("[WARN] Haze model load failed:", e)


# ============================================================
# 3. WAQI NUMERIC
# ============================================================
city = "Delhi"
waqi_url = f"https://api.waqi.info/feed/{city}/?token={WAQI_TOKEN}"

waqi_json = requests.get(waqi_url).json()
waqi_aqi = float(waqi_json["data"]["aqi"])

num_sigma = 25.0  # validated numeric baseline

print("\nWAQI AQI:", waqi_aqi, "sigma:", num_sigma)


# ============================================================
# 4. NEWS (OpenWeather)
# ============================================================
weather_url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={OPENWEATHER_KEY}"
weather_json = requests.get(weather_url).json()

headline = weather_json.get("weather", [{}])[0].get("description", "")
embedding = [0.31, 0.22, 0.18, 0.44, 0.29, 0.41, 0.17, 0.28, 0.26, 0.33]  # heuristic

aqi_txt, conf_txt, tags = predict_from_news(headline, embedding)

# convert confidence → sigma
news_sigma = 40 * (1 - conf_txt)
if news_sigma < 8:
    news_sigma = 8.0  # safety floor

print("\nNEWS HEADLINE:", headline)
print("NEWS AQI:", aqi_txt, "conf:", conf_txt, "sigma:", news_sigma)
print("NEWS TAGS:", tags)


# ============================================================
# 5. LITEFUSION — Inverse Variance Fusion
# ============================================================
core = LiteFusionCore()

estimates = [
    ModalityEstimate(name="image",   mean=img_aqi,  sigma=img_sigma),
    ModalityEstimate(name="numeric", mean=waqi_aqi, sigma=num_sigma),
    ModalityEstimate(name="news",    mean=aqi_txt,  sigma=news_sigma),
]

fused = core.fuse(estimates)

print("\n=============================")
print("FINAL LITEFUSION AQI:", fused.mean)
print("sigma:", fused.sigma)
print("weights:", fused.norm_weights)
print("=============================\n")

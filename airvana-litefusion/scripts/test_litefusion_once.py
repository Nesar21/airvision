import os, sys, numpy as np

# ------------------ ENV SETUP ------------------
from dotenv import load_dotenv

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

print("WAQI_TOKEN:", os.getenv("WAQI_TOKEN"))
print("OPENWEATHER_KEY:", os.getenv("OPENWEATHER_KEY"))

SRC = os.path.join(PROJECT_ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

# ------- IMPORTS AFTER PATH FIX -------
from image_model.mobilenet_predictor import MobileNetAQIPredictor
from fusion.litefusion_core import LiteFusionPredictor

# ------------------ IMAGE ------------------
model_paths = [
    "models/mobilenet/mnv3_fold0.pt",
    "models/mobilenet/mnv3_fold1.pt",
    "models/mobilenet/mnv3_fold2.pt",
    "models/mobilenet/mnv3_fold3.pt",
    "models/mobilenet/mnv3_fold4.pt"
]

img_path = "data/images/manual_test/Delhi-Air-Pollution-1-1.jpg"

preds = []
for mp in model_paths:
    print("Loading:", mp)
    model = MobileNetAQIPredictor(mp)
    p = model.predict(img_path)
    preds.append(float(p))

aqi_img = float(np.mean(preds))

print("\n--- IMAGE-ONLY PREDICTION ---")
print("Per-model predictions:", preds)
print("Image-only AQI:", round(aqi_img, 2))

# ----------------- WAQI -------------------
WAQI_TOKEN = os.getenv("WAQI_TOKEN")
if WAQI_TOKEN:
    import requests
    url = f"https://api.waqi.info/feed/delhi/?token={WAQI_TOKEN}"
    wq = requests.get(url).json()
    aqi_wq = float(wq["data"]["aqi"])
    print("WAQI AQI:", aqi_wq)
else:
    aqi_wq = None
    print("WAQI_TOKEN not set — skipping WAQI.")

# ----------------- OpenWeather -------------------
OPENWEATHER_KEY = os.getenv("OPENWEATHER_KEY")
if OPENWEATHER_KEY:
    import requests
    url = f"https://api.openweathermap.org/data/2.5/air_pollution?lat=28.7041&lon=77.1025&appid={OPENWEATHER_KEY}"
    ow = requests.get(url).json()
    open_aqi = ow["list"][0]["main"]["aqi"] * 100
    print("OpenWeather AQI:", open_aqi)
else:
    open_aqi = None
    print("OPENWEATHER_KEY not set — skipping OpenWeather.")

# ----------------- FUSION -------------------
fusion = LiteFusionPredictor()

aqi_fused = fusion.fuse_aqi(
    image=(aqi_img, 8.0),
    numeric=(aqi_wq, 25.0) if aqi_wq else None,
    news=(open_aqi, 8.0) if open_aqi else None
).mean

# ----------------- RESULTS -------------------
print("\n========== FINAL RESULT ==========")
print("Image-only AQI:", aqi_img)
print("WAQI AQI:", aqi_wq)
print("OpenWeather AQI:", open_aqi)
print("LiteFusion AQI:", aqi_fused)
print("=================================")

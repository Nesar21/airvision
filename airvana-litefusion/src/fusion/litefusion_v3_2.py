# src/fusion/litefusion_v3_2.py
"""
LiteFusion V3.2 (stable)
- Image + WAQI + News fusion
- Distance-weighted WAQI sigma
- News cross-validation (prevents false spikes)
- OpenWeather structured delta (weather_main / visibility / wind)
- Robust prediction_from_news output handling (tuple or dict)
- Minimal, no Gemini, no embeddings
"""

from __future__ import annotations
import os, math, json
from typing import Optional, Tuple, Dict, Any

# ------------------------------
# IMPORT MODELS
# ------------------------------
try:
    from src.image_model.mobilenet_predictor import MobileNetAQIPredictor
except:
    from image_model.mobilenet_predictor import MobileNetAQIPredictor

try:
    from src.text_model.text_predictor import predict_from_news
except:
    from text_model.text_predictor import predict_from_news


# ------------------------------
# CONSTANTS
# ------------------------------
EPS = 1e-6

BASE_WAQI_SIGMA = 15.0
BASE_NEWS_SIGMA = 25.0
IMG_SIGMA_FLOOR = 8.0

DISTANCE_THRESHOLD_KM = 10.0
DISTANCE_SCALE_KM = 20.0

CONFLICT_THRESH = 40.0
CONFLICT_SMOOTH = 10.0


# ------------------------------
# UTILITIES
# ------------------------------
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = map(math.radians, (lat1, lat2))
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dl/2)**2
    return 2 * R * math.asin(math.sqrt(a))


def sigmoid_weight(gap, threshold=CONFLICT_THRESH, smooth=CONFLICT_SMOOTH):
    return 1.0 / (1.0 + math.exp(-(gap - threshold)/smooth))


def inverse_variance_fuse(mods: Tuple[Tuple[float, float], ...]):
    wsum = 0.0
    ms = 0.0
    for mu, sigma in mods:
        w = 1.0 / (sigma*sigma + EPS)
        wsum += w
        ms += w * mu
    mu = ms / (wsum + EPS)
    sigma = 1.0 / math.sqrt(wsum + EPS)
    return mu, sigma


# ------------------------------
# LITEFUSION CLASS
# ------------------------------
class LiteFusionV3_2:
    def __init__(self, model_path: str, use_blur_gate=False, blur_threshold=100.0):
        self.use_blur_gate = use_blur_gate
        self.blur_threshold = blur_threshold
        self.img_predictor = MobileNetAQIPredictor(model_path=model_path)

    # -----------------------------------------
    # IMAGE ESTIMATE
    # -----------------------------------------
    def compute_image_estimate(self, image_path, raw_image=None):
        meta = {}
        out = None

        try:
            out = self.img_predictor.predict(image_path, return_uncertainty=True, raw_image=raw_image)
        except TypeError:
            out = self.img_predictor.predict(image_path)

        # handle dict
        if isinstance(out, dict):
            mu = float(out.get("aqi") or out.get("mu") or out.get("mu_img") or out.get("value"))
            sigma = float(out.get("sigma") or out.get("uncertainty") or out.get("std", IMG_SIGMA_FLOOR))
            meta.update(out)

        # handle tuple/list
        elif isinstance(out, (list, tuple)):
            mu = float(out[0])
            sigma = float(out[1]) if len(out) > 1 else IMG_SIGMA_FLOOR
            meta["raw_tuple"] = out

        # handle raw float
        else:
            mu = float(out)
            sigma = IMG_SIGMA_FLOOR
            meta["raw_output"] = out

        # blur-gate (optional)
        if self.use_blur_gate and raw_image is not None:
            try:
                import cv2
                gray = cv2.cvtColor(raw_image, cv2.COLOR_BGR2GRAY)
                lap = cv2.Laplacian(gray, cv2.CV_64F).var()
                meta["laplacian_var"] = float(lap)
                if lap < self.blur_threshold:
                    sigma *= 3.0
                    meta["blur_penalty"] = True
            except:
                pass

        return mu, sigma, meta

    # -----------------------------------------
    # WAQI SIGMA WITH DISTANCE
    # -----------------------------------------
    def compute_waqi_sigma(self, waqi, ulat, ulon, slat, slon):
        sigma = BASE_WAQI_SIGMA
        if None not in (ulat, ulon, slat, slon):
            d = haversine_km(ulat, ulon, slat, slon)
            if d > DISTANCE_THRESHOLD_KM:
                sigma *= 1.0 + (d / DISTANCE_SCALE_KM)
        return sigma

    # -----------------------------------------
    # OPENWEATHER STRUCTURED DELTA
    # -----------------------------------------
    def openweather_delta(self, weather_main, visibility, wind_speed):
        delta = 0.0
        conf_mod = 1.0

        if not weather_main:
            return delta, conf_mod

        wm = weather_main.lower()
        if "smoke" in wm:
            delta += 25
            conf_mod += 0.2
        if "haze" in wm:
            delta += 15
            conf_mod += 0.15
        if "dust" in wm:
            delta += 15
            conf_mod += 0.1
        if "fog" in wm:
            delta -= 5

        # visibility
        if visibility is not None:
            try:
                if visibility < 2000:
                    delta += 10
                    conf_mod += 0.1
            except:
                pass

        # wind
        if wind_speed and wind_speed > 8:
            delta *= 0.7
            conf_mod *= 0.9

        return delta, max(0.1, min(2.0, conf_mod))

    # -----------------------------------------
    # NEWS CROSS-VALIDATION
    # -----------------------------------------
    def validate_news_delta(self, delta_news, mu_img, mu_waqi):
        if delta_news > 20 and mu_waqi is not None:
            if mu_img < mu_waqi - 10:
                return 0.0
        return delta_news

    # -----------------------------------------
    # ADAPTIVE SIGMA (conflict resolution)
    # -----------------------------------------
    def adaptive_sigma(self, mu_img, mu_waqi, sigma_img, sigma_waqi):
        if mu_waqi is None:
            return sigma_img, sigma_waqi

        gap = abs(mu_img - mu_waqi)
        w = sigmoid_weight(gap)
        sigma_waqi *= (1 + 2*w)

        if sigma_img > 15:
            sigma_img *= 2.5

        return sigma_img, sigma_waqi

    # -----------------------------------------
    # FUSION
    # -----------------------------------------
    def fuse(
        self,
        image_path: str,
        waqi_aqi: Optional[float] = None,
        user_lat: Optional[float] = None,
        user_lon: Optional[float] = None,
        station_lat: Optional[float] = None,
        station_lon: Optional[float] = None,
        news_text: Optional[str] = None,
        weather_main: Optional[str] = None,
        visibility: Optional[float] = None,
        wind_speed: Optional[float] = None,
        raw_image=None,
        include_news=True,
        include_waqi=True
    ):

        # 1) IMAGE
        mu_img, sigma_img, img_meta = self.compute_image_estimate(image_path, raw_image)

        # 2) WAQI
        mu_waqi = float(waqi_aqi) if waqi_aqi is not None else None
        sigma_waqi = self.compute_waqi_sigma(mu_waqi, user_lat, user_lon, station_lat, station_lon) if mu_waqi else 100

        # 3) OPENWEATHER DELTA
        ow_delta, ow_conf = self.openweather_delta(weather_main, visibility, wind_speed)

        # 4) NEWS DELTA (dict / tuple safe)
        delta_news = 0.0
        sigma_news = BASE_NEWS_SIGMA
        news_meta = {}

        if include_news and news_text:
            pred = predict_from_news(news_text, None)

            # HANDLE BOTH DICT + TUPLE
            if isinstance(pred, dict):
                delta_news = float(pred.get("aqi_txt", 0.0))
                conf = float(pred.get("confidence", 1.0))
                news_meta.update(pred)

            elif isinstance(pred, (tuple, list)):
                # (aqi_txt, confidence, ...)
                try:
                    delta_news = float(pred[0])
                except:
                    delta_news = 0.0
                try:
                    conf = float(pred[1])
                except:
                    conf = 1.0
                news_meta = {"raw_tuple": pred, "aqi_txt": delta_news, "confidence": conf}

            else:
                try:
                    delta_news = float(pred)
                except:
                    delta_news = 0.0
                conf = 1.0
                news_meta = {"raw_output": pred}

            # merge with OW
            if ow_delta != 0:
                if (ow_delta > 0 and delta_news > 0) or (ow_delta < 0 and delta_news < 0):
                    delta_news = delta_news + 0.6 * ow_delta
                else:
                    delta_news = delta_news * 0.6 + 0.4 * ow_delta

            # inverse confidence weighting
            sigma_news = max(BASE_NEWS_SIGMA, (30.0 / max(conf, 0.01))) / ow_conf

            # cross-validation
            delta_news = self.validate_news_delta(delta_news, mu_img, mu_waqi)

        else:
            # news absent → use OW only
            delta_news = ow_delta
            sigma_news = max(BASE_NEWS_SIGMA, BASE_NEWS_SIGMA / max(ow_conf, 0.1))

        # 5) ADAPTIVE SIGMA
        sigma_img, sigma_waqi = self.adaptive_sigma(mu_img, mu_waqi, sigma_img, sigma_waqi)

        # 6) BUILD MODALITIES
        modalities = []
        if include_waqi and mu_waqi is not None:
            modalities.append((mu_waqi, sigma_waqi))

        modalities.append((mu_img, sigma_img))

        ref = mu_waqi if mu_waqi is not None else mu_img
        mu_news = ref + delta_news

        if include_news:
            modalities.append((mu_news, sigma_news))

        # 7) FUSE
        mu_fused, sigma_fused = inverse_variance_fuse(tuple(modalities))

        return {
            "mu_img": mu_img, "sigma_img": sigma_img,
            "mu_waqi": mu_waqi, "sigma_waqi": sigma_waqi,
            "delta_news": delta_news, "mu_news": mu_news, "sigma_news": sigma_news,
            "mu_fused": mu_fused, "sigma_fused": sigma_fused,
            "img_meta": img_meta, "news_meta": news_meta,
            "modalities_used": [("waqi", mu_waqi is not None), ("image", True), ("news", include_news)]
        }


# ----------------------------------------------------
# CLI
# ----------------------------------------------------
def run_cli():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--waqi", type=float)
    parser.add_argument("--user-lat", type=float)
    parser.add_argument("--user-lon", type=float)
    parser.add_argument("--station-lat", type=float)
    parser.add_argument("--station-lon", type=float)
    parser.add_argument("--news", type=str)
    parser.add_argument("--weather-main", type=str)
    parser.add_argument("--visibility", type=float)
    parser.add_argument("--wind-speed", type=float)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--use-blur-gate", action="store_true")
    args = parser.parse_args()

    lf = LiteFusionV3_2(args.model_path, use_blur_gate=args.use_blur_gate)
    out = lf.fuse(
        image_path=args.image,
        waqi_aqi=args.waqi,
        user_lat=args.user_lat,
        user_lon=args.user_lon,
        station_lat=args.station_lat,
        station_lon=args.station_lon,
        news_text=args.news,
        weather_main=args.weather_main,
        visibility=args.visibility,
        wind_speed=args.wind_speed
    )
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    run_cli()

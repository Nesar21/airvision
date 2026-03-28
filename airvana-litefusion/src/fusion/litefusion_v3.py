# src/fusion/litefusion_v3.py
"""
LiteFusion 3.0
- WAQI anchor
- Image confidence gate
- News delta (with validation)
- Adaptive sigma conflict handling
- Optional distance-based WAQI weighting (auto-enable when station > DIST_THRESHOLD)
- Minimal dependencies; integrates with existing project predictors.
"""

from __future__ import annotations
import os
import math
import json
from typing import Optional, Tuple, Dict, Any

# Attempt to import your repo modules (adjust paths via PYTHONPATH if needed)
try:
    from src.image_model.mobilenet_predictor import MobileNetAQIPredictor
except Exception:
    # try relative import for direct execution
    from image_model.mobilenet_predictor import MobileNetAQIPredictor

try:
    from src.text_model.text_predictor import predict_from_news
except Exception:
    from text_model.text_predictor import predict_from_news

# Constants / defaults
EPS = 1e-6
BASE_WAQI_SIGMA = 15.0
BASE_NEWS_SIGMA = 25.0
IMG_SIGMA_FLOOR = 8.0
DISTANCE_THRESHOLD_KM = 10.0  # start applying distance weighting after this
DISTANCE_SCALE = 20.0         # used in sigma multiplier: 1 + distance_km/DISTANCE_SCALE
CONFLICT_THRESH = 40.0
CONFLICT_SMOOTH = 10.0

# Utility functions
def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return 2 * R * math.asin(math.sqrt(a))

def sigmoid_weight(gap: float, threshold: float = CONFLICT_THRESH, smooth: float = CONFLICT_SMOOTH) -> float:
    return 1.0 / (1.0 + math.exp(-(gap - threshold) / smooth))

# Optional image quality helper (lightweight). Keep optional (off by default).
def laplacian_variance_gray(image_bgr) -> float:
    """A fast laplacian var check. Accepts BGR numpy image (cv2)."""
    try:
        import cv2
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())
    except Exception:
        return float("nan")

# Core fusion utilities
def inverse_variance_fuse(modalities: Tuple[Tuple[float, float], ...]) -> Tuple[float, float]:
    """Fuse modalities given (mean, sigma) tuples using inverse-variance weighting."""
    weights = []
    weighted_sum = 0.0
    for mu, sigma in modalities:
        w = 1.0 / ((sigma ** 2) + EPS)
        weights.append(w)
        weighted_sum += w * mu
    wsum = sum(weights) + EPS
    mu_fused = weighted_sum / wsum
    sigma_fused = 1.0 / math.sqrt(wsum)
    return mu_fused, sigma_fused

# High-level pipeline
class LiteFusionV3:
    def __init__(
        self,
        mobilenet_model_path: Optional[str] = None,
        use_blur_gate: bool = False,
        blur_threshold: float = 100.0,
        distance_threshold_km: float = DISTANCE_THRESHOLD_KM
    ):
        # instantiate image predictor (expects existing class)
        self.img_predictor = MobileNetAQIPredictor(model_path=mobilenet_model_path) if MobileNetAQIPredictor else None
        self.use_blur_gate = use_blur_gate
        self.blur_threshold = blur_threshold
        self.distance_threshold_km = distance_threshold_km

    def compute_image_estimate(self, image_path: str, raw_image=None) -> Tuple[float, float, Dict[str, Any]]:
        """
        Returns (mu_img, sigma_img, meta) where meta includes image_quality if available.
        If your MobileNet predictor returns sigma/confidence, use it; else derive sigma from validation MAE or floor.
        """
        meta = {}
        # prefer using predictor API if it provides both mu and sigma
        mu_img = None
        sigma_img = None
        if self.img_predictor:
            out = self.img_predictor.predict(image_path, return_uncertainty=True)
            # expected: {'aqi': float, 'sigma': float, ...} or (aqi, sigma)
            if isinstance(out, dict):
                mu_img = float(out.get("aqi"))
                sigma_img = float(out.get("sigma", out.get("uncertainty", IMG_SIGMA_FLOOR)))
                meta.update(out)
            elif isinstance(out, (tuple, list)) and len(out) >= 1:
                mu_img = float(out[0])
                sigma_img = float(out[1]) if len(out) >= 2 else IMG_SIGMA_FLOOR

        # fallbacks if predictor unavailable/doesn't return sigma
        if mu_img is None:
            # try a minimal loader (so script doesn't crash). Attempt to read image and call predictor's simpler API
            mu_img = float(self.img_predictor.predict(image_path)) if self.img_predictor else 50.0
        if sigma_img is None:
            # use predictor val MAE if available in file (common pattern: mnv3_fold_metrics), else floor
            sigma_img = max(self._read_image_val_mae(), IMG_SIGMA_FLOOR)

        # optional blur gate check (if user enabled)
        image_quality = None
        if self.use_blur_gate:
            try:
                import cv2
                if raw_image is None:
                    raw_image = cv2.imread(image_path)
                blur_score = laplacian_variance_gray(raw_image)
                image_quality = {"blur_score": blur_score}
                if not math.isnan(blur_score) and blur_score < self.blur_threshold:
                    # increase sigma (reduce trust) for blurry images
                    sigma_img *= 3.0
                    image_quality["blur_penalized"] = True
            except Exception:
                pass

        meta["mu_img"] = mu_img
        meta["sigma_img"] = sigma_img
        if image_quality:
            meta.update(image_quality)
        return mu_img, sigma_img, meta

    def _read_image_val_mae(self) -> float:
        # try to read stored mobilenet fold metrics in repo
        try:
            repo_metrics = os.path.join("models", "mobilenet", "mnv3_fold_metrics.txt")
            if os.path.exists(repo_metrics):
                with open(repo_metrics, "r") as f:
                    txt = f.read()
                # naive parse: look for "MAE: 12.3" or similar
                import re
                m = re.search(r"MAE[:=]\s*([0-9]*\.?[0-9]+)", txt)
                if m:
                    return float(m.group(1))
        except Exception:
            pass
        return IMG_SIGMA_FLOOR

    def compute_waqi_sigma(self, waqi_aqi: float, user_lat: Optional[float], user_lon: Optional[float],
                           station_lat: Optional[float], station_lon: Optional[float]) -> float:
        sigma = BASE_WAQI_SIGMA
        # optional distance weighting
        if user_lat is not None and user_lon is not None and station_lat is not None and station_lon is not None:
            dist_km = haversine_km(user_lat, user_lon, station_lat, station_lon)
            if dist_km > self.distance_threshold_km:
                sigma *= (1.0 + (dist_km / DISTANCE_SCALE))
        return sigma

    def validate_news_delta(self, delta_news: float, mu_img: float, mu_waqi: float) -> float:
        # news validation: if news claims large pollution but image indicates cleaner than WAQI, ignore delta
        if delta_news is None:
            return 0.0
        try:
            if delta_news > 20.0 and mu_img < mu_waqi - 10.0:
                return 0.0
        except Exception:
            pass
        return delta_news

    def adaptive_sigma_adjust(self, mu_img: float, mu_waqi: float, sigma_img: float, sigma_waqi: float) -> Tuple[float, float]:
        gap = abs(mu_img - mu_waqi)
        conf_w = sigmoid_weight(gap)
        # increase waqi sigma smoothly with gap
        sigma_waqi = sigma_waqi * (1.0 + 2.0 * conf_w)
        # adjust image sigma if it is large (low confidence)
        if sigma_img > 15.0:
            sigma_img = sigma_img * 2.5
        return sigma_img, sigma_waqi

    def fuse(
        self,
        image_path: str,
        waqi_aqi: float,
        user_lat: Optional[float] = None,
        user_lon: Optional[float] = None,
        station_lat: Optional[float] = None,
        station_lon: Optional[float] = None,
        news_text: Optional[str] = None,
        raw_image=None,
        include_news: bool = True,
        include_waqi: bool = True
    ) -> Dict[str, Any]:
        """
        Main entrypoint.
        Returns dict: {mu_img, sigma_img, mu_waqi, sigma_waqi, delta_news, mu_news, sigma_news, mu_fused, sigma_fused, meta...}
        """
        # 1. image
        mu_img, sigma_img, img_meta = self.compute_image_estimate(image_path, raw_image=raw_image)

        # 2. waqi
        mu_waqi = float(waqi_aqi) if waqi_aqi is not None else None
        sigma_waqi = self.compute_waqi_sigma(mu_waqi, user_lat, user_lon, station_lat, station_lon) if mu_waqi is not None else 100.0

        # 3. news
        delta_news = 0.0
        news_meta = {}
        if include_news and news_text:
            # predict_from_news returns (delta, conf, tags) or dict
            pred = predict_from_news(news_text)  # keep existing function interface
            if isinstance(pred, dict):
                delta_news = float(pred.get("aqi_txt", pred.get("delta", 0.0)))
                conf = float(pred.get("confidence", 1.0))
                news_meta.update(pred)
            elif isinstance(pred, (list, tuple)):
                delta_news = float(pred[0])
                conf = float(pred[1]) if len(pred) > 1 else 1.0
            else:
                delta_news = float(pred)
                conf = 1.0
            # optional: convert confidence to sigma for news (not strictly required)
            sigma_news = max(BASE_NEWS_SIGMA, (30.0 / max(conf, 0.01)))
            # validate delta
            delta_news = self.validate_news_delta(delta_news, mu_img, mu_waqi if mu_waqi is not None else mu_img)
        else:
            delta_news = 0.0
            sigma_news = BASE_NEWS_SIGMA

        mu_news = (mu_waqi if mu_waqi is not None else mu_img) + delta_news

        # 4. adaptive sigma adjustments
        if mu_waqi is None:
            # no WAQI available: fuse image + news (news is delta added to image)
            sigma_img = sigma_img
            sigma_news = sigma_news
        else:
            sigma_img, sigma_waqi = self.adaptive_sigma_adjust(mu_img, mu_waqi, sigma_img, sigma_waqi)

        # 5. assemble modalities (order: waqi, image, news)
        modalities = []
        if include_waqi and mu_waqi is not None:
            modalities.append((mu_waqi, sigma_waqi))
        # always include image
        modalities.append((mu_img, sigma_img))
        # include news as adjusted mean
        if include_news:
            modalities.append((mu_news, sigma_news))

        mu_fused, sigma_fused = inverse_variance_fuse(tuple(modalities))

        # results
        out = {
            "mu_img": mu_img,
            "sigma_img": sigma_img,
            "mu_waqi": mu_waqi,
            "sigma_waqi": sigma_waqi,
            "delta_news": delta_news,
            "mu_news": mu_news,
            "sigma_news": sigma_news,
            "mu_fused": mu_fused,
            "sigma_fused": sigma_fused,
            "modalities_used": [("waqi", mu_waqi is not None and include_waqi), ("image", True), ("news", include_news)],
            "img_meta": img_meta,
            "news_meta": news_meta,
        }
        return out

# Simple CLI for manual runs
def run_cli():
    import argparse
    parser = argparse.ArgumentParser(description="LiteFusion V3 runner")
    parser.add_argument("--image", required=True, help="path to image")
    parser.add_argument("--waqi", type=float, required=False, help="WAQI numeric value (if available)")
    parser.add_argument("--user-lat", type=float, required=False)
    parser.add_argument("--user-lon", type=float, required=False)
    parser.add_argument("--station-lat", type=float, required=False)
    parser.add_argument("--station-lon", type=float, required=False)
    parser.add_argument("--news", type=str, required=False, help="news text / weather description")
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--use-blur-gate", action="store_true")
    args = parser.parse_args()

    lf = LiteFusionV3(mobilenet_model_path=args.model_path, use_blur_gate=args.use_blur_gate)
    res = lf.fuse(
        image_path=args.image,
        waqi_aqi=args.waqi,
        user_lat=args.user_lat,
        user_lon=args.user_lon,
        station_lat=args.station_lat,
        station_lon=args.station_lon,
        news_text=args.news,
        include_news=bool(args.news),
        include_waqi=(args.waqi is not None)
    )
    print(json.dumps(res, indent=2))

if __name__ == "__main__":
    run_cli()

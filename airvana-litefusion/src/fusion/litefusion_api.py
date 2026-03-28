#!/usr/bin/env python3
"""
litefusion_api.py

High-level API for LiteFusion.

This wraps LiteFusionCore and provides a clean interface to fuse:
- image-based AQI (or PM2.5) predictions
- weather / numeric model predictions
- news tone predictions
- optional extra modalities in the future

Usage pattern in your pipeline:

    from fusion.litefusion_core import LiteFusionCore, ModalityEstimate
    from fusion.litefusion_api import LiteFusionPredictor

    fusion = LiteFusionPredictor()

    # Example values coming from your models:
    aqi_img,  unc_img  = 130.0, 8.0
    aqi_num,  unc_num  = 110.0, 15.0
    aqi_txt,  conf_txt = 20.0, 0.6   # from text_predictor

    fused = fusion.fuse_aqi(
        image=(aqi_img, unc_img),
        numeric=(aqi_num, unc_num),
        news=(aqi_txt, conf_txt),
    )

    print(fused.mean, fused.sigma, fused.norm_weights)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict

from .litefusion_core import LiteFusionCore, ModalityEstimate, FusedResult


@dataclass
class AQIModalityInputs:
    """
    Structured container for AQI modalities (optional).
    """

    image: Optional[Tuple[float, float]] = None     # (mean, sigma)
    numeric: Optional[Tuple[float, float]] = None   # (mean, sigma)
    news: Optional[Tuple[float, float]] = None      # (mean, confidence)
    satellite: Optional[Tuple[float, float]] = None # (mean, sigma)
    pm25_to_aqi: Optional[Tuple[float, float]] = None  # (mean, sigma)


class LiteFusionPredictor:
    """
    High-level LiteFusion API.

    Responsibilities:
    - convert various modality outputs into ModalityEstimate
    - map news confidence → sigma
    - call LiteFusionCore.fuse()
    - return FusedResult for AQI (or other scalar)
    """

    def __init__(
        self,
        # base sigmas (can be tuned later)
        base_sigma_image: float = 8.0,
        base_sigma_numeric: float = 12.0,
        base_sigma_news: float = 30.0,
        base_sigma_satellite: float = 20.0,
        base_sigma_pm25aqi: float = 10.0,
    ) -> None:
        self.core = LiteFusionCore()
        self.base_sigma_image = float(base_sigma_image)
        self.base_sigma_numeric = float(base_sigma_numeric)
        self.base_sigma_news = float(base_sigma_news)
        self.base_sigma_satellite = float(base_sigma_satellite)
        self.base_sigma_pm25aqi = float(base_sigma_pm25aqi)

    # ----------------------------
    # helpers
    # ----------------------------

    @staticmethod
    def _clamp_conf(conf: float) -> float:
        if conf is None:
            return 0.0
        if conf < 0.0:
            return 0.0
        if conf > 1.0:
            return 1.0
        return float(conf)

    def _sigma_from_conf(self, base_sigma: float, confidence: float) -> float:
        """
        Convert confidence in [0,1] → sigma.

        High confidence → smaller sigma → higher weight.

        Map:
            sigma = base_sigma / (0.3 + 0.7 * confidence)
        so that:
            conf = 0.0 → ~3.33 * base_sigma
            conf = 1.0 → 1.0 * base_sigma
        """
        c = self._clamp_conf(confidence)
        scale = 0.3 + 0.7 * c
        return base_sigma / scale

    # ----------------------------
    # main AQI fusion entry point
    # ----------------------------

    def fuse_aqi(
        self,
        *,
        image: Optional[Tuple[float, float]] = None,      # (aqi_img, sigma_img)
        numeric: Optional[Tuple[float, float]] = None,    # (aqi_num, sigma_num)
        news: Optional[Tuple[float, float]] = None,       # (aqi_txt, confidence_txt)
        satellite: Optional[Tuple[float, float]] = None,  # (aqi_sat, sigma_sat)
        pm25_to_aqi: Optional[Tuple[float, float]] = None # (aqi_from_pm25, sigma_pm25aqi)
    ) -> Optional[FusedResult]:
        """
        Fuse AQI signals from multiple modalities.

        All arguments are optional. Only provided ones will be used.
        """
        estimates = []

        # Image modality
        if image is not None:
            aqi_img, sigma_img = image
            # if sigma not provided or <=0, fall back to base
            if sigma_img is None or sigma_img <= 0.0:
                sigma_img = self.base_sigma_image
            estimates.append(
                ModalityEstimate("image", float(aqi_img), float(sigma_img))
            )

        # Numeric (weather + pollution) modality
        if numeric is not None:
            aqi_num, sigma_num = numeric
            if sigma_num is None or sigma_num <= 0.0:
                sigma_num = self.base_sigma_numeric
            estimates.append(
                ModalityEstimate("numeric", float(aqi_num), float(sigma_num))
            )

        # Satellite-derived AQI modality
        if satellite is not None:
            aqi_sat, sigma_sat = satellite
            if sigma_sat is None or sigma_sat <= 0.0:
                sigma_sat = self.base_sigma_satellite
            estimates.append(
                ModalityEstimate("satellite", float(aqi_sat), float(sigma_sat))
            )

        # AQI derived indirectly from PM2.5 vision models
        if pm25_to_aqi is not None:
            aqi_pm, sigma_pm = pm25_to_aqi
            if sigma_pm is None or sigma_pm <= 0.0:
                sigma_pm = self.base_sigma_pm25aqi
            estimates.append(
                ModalityEstimate("pm25_vision", float(aqi_pm), float(sigma_pm))
            )

        # News modality (aqi_txt + confidence_txt)
        if news is not None:
            aqi_txt, conf_txt = news
            sigma_news = self._sigma_from_conf(self.base_sigma_news, conf_txt)
            estimates.append(
                ModalityEstimate("news", float(aqi_txt), float(sigma_news))
            )

        if not estimates:
            return None

        return self.core.fuse(estimates)


# --------------------------
# Simple CLI sanity test
# --------------------------
if __name__ == "__main__":
    fusion = LiteFusionPredictor()

    # Example: image and weather dominate, news slightly pushes up/down
    aqi_img, unc_img = 135.0, 8.0
    aqi_num, unc_num = 120.0, 15.0
    aqi_txt, conf_txt = 20.0, 0.7  # news says "smog, fire" → positive tone

    fused = fusion.fuse_aqi(
        image=(aqi_img, unc_img),
        numeric=(aqi_num, unc_num),
        news=(aqi_txt, conf_txt),
    )

    print("Fused AQI:", fused.mean)
    print("Fused sigma:", fused.sigma)
    print("Normalized weights:", fused.norm_weights)

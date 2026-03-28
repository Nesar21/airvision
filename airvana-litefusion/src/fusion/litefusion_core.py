#!/usr/bin/env python3
"""
litefusion_core.py

Variance-based fusion of multiple AQI (or PM2.5) estimates.

Implements the core math:

    w_i        = 1 / (sigma_i**2 + eps)
    mu_fused   = Σ_i(w_i * mu_i) / Σ_i(w_i)
    sigma_fused = 1 / sqrt(Σ_i(w_i))

Where:
- mu_i    : prediction from modality i (e.g., image, weather, news)
- sigma_i : uncertainty (std dev) of modality i

This is the "LiteFusion" core that you can describe as:
- probabilistic inverse-variance weighting
- uncertainty-aware model fusion

No deep nets here, just clean math.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
import math


@dataclass
class ModalityEstimate:
    """
    Single modality estimate.

    name   : identifier, e.g. "image", "pm25_model", "news", "weather"
    mean   : predicted value (AQI, PM2.5, etc.)
    sigma  : std dev estimate of this prediction
    enabled: if False, modality is ignored
    """
    name: str
    mean: float
    sigma: float
    enabled: bool = True


@dataclass
class FusedResult:
    """
    Output of LiteFusion.

    mean         : fused prediction
    sigma        : fused uncertainty (std dev)
    weights      : raw weights w_i for each modality
    norm_weights : normalized weights (w_i / Σ w_i) for explainability
    """
    mean: float
    sigma: float
    weights: Dict[str, float]
    norm_weights: Dict[str, float]


class LiteFusionCore:
    """
    Core LiteFusion engine.

    eps        : numerical stability in 1/(sigma^2 + eps)
    min_sigma  : lower bound for sigma to avoid infinite weights
    max_sigma  : upper bound for sigma so out-of-range estimates don't explode
    """

    def __init__(
        self,
        eps: float = 1e-6,
        min_sigma: float = 1.0,
        max_sigma: float = 80.0,
    ) -> None:
        self.eps = float(eps)
        self.min_sigma = float(min_sigma)
        self.max_sigma = float(max_sigma)

    def _sanitize_sigma(self, sigma: float) -> float:
        """
        Clamp and fix weird sigma values.

        - None or non-finite → max_sigma (very low trust)
        - below min_sigma     → min_sigma
        - above max_sigma     → max_sigma
        """
        if sigma is None or not math.isfinite(sigma):
            return self.max_sigma
        sigma = float(sigma)
        if sigma < self.min_sigma:
            return self.min_sigma
        if sigma > self.max_sigma:
            return self.max_sigma
        return sigma

    def fuse(self, estimates: List[ModalityEstimate]) -> Optional[FusedResult]:
        """
        Fuse multiple modality estimates into one.

        Args:
            estimates: list of ModalityEstimate

        Returns:
            FusedResult or None if no enabled modalities.
        """
        # keep only enabled
        active = [e for e in estimates if e.enabled]
        if not active:
            return None

        # compute weights
        weight_map: Dict[str, float] = {}
        for est in active:
            s = self._sanitize_sigma(est.sigma)
            w = 1.0 / (s * s + self.eps)
            weight_map[est.name] = w

        # compute fused mean
        num = 0.0
        denom = 0.0
        for est in active:
            w = weight_map[est.name]
            num += w * float(est.mean)
            denom += w

        if denom <= 0.0:
            # fallback (should not really happen if min_sigma > 0)
            # use simple average of means
            simple = sum(float(e.mean) for e in active) / len(active)
            return FusedResult(
                mean=simple,
                sigma=self.max_sigma,
                weights=weight_map,
                norm_weights={k: 1.0 / len(weight_map) for k in weight_map},
            )

        mean_fused = num / denom
        sigma_fused = 1.0 / math.sqrt(denom)

        # normalized weights for explainability
        wsum = sum(weight_map.values())
        if wsum <= 0.0:
            norm = {k: 1.0 / len(weight_map) for k in weight_map}
        else:
            norm = {k: w / wsum for k, w in weight_map.items()}

        return FusedResult(
            mean=float(mean_fused),
            sigma=float(sigma_fused),
            weights=weight_map,
            norm_weights=norm,
        )


# --------------------------
# Quick CLI sanity test
# --------------------------
if __name__ == "__main__":
    # Example: image is confident, weather and news are weak
    image_est = ModalityEstimate(name="image", mean=120.0, sigma=8.0)
    weather_est = ModalityEstimate(name="weather", mean=90.0, sigma=20.0)
    news_est = ModalityEstimate(name="news", mean=30.0, sigma=40.0)

    core = LiteFusionCore()
    fused = core.fuse([image_est, weather_est, news_est])

    print("Fused mean:", fused.mean)
    print("Fused sigma:", fused.sigma)
    print("Raw weights:", fused.weights)
    print("Normalized weights:", fused.norm_weights)

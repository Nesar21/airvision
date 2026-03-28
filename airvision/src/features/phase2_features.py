from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.utils.env_utils import init_env
from src.utils.manifest import log_run


# ----------------------------------------------------
#  LOW-LEVEL IMAGE FEATURE EXTRACTOR
# ----------------------------------------------------


def _compute_dark_channel_prior(img: np.ndarray, patch_size: int = 15) -> float:
    """
    Scalar dark channel prior value.

    Steps:
        - For each pixel: take min over RGB channels
        - Apply min-filter (erosion) with patch_size
        - Return mean of the dark channel
    """
    if img.ndim != 3:
        return np.nan

    dark = np.min(img, axis=2).astype(np.uint8)
    kernel = np.ones((patch_size, patch_size), np.uint8)
    dark_eroded = cv2.erode(dark, kernel)
    return float(dark_eroded.mean())


def _compute_sky_fraction(img: np.ndarray) -> float:
    """
    Very simple sky heuristic:
        - Take the top 1/3 of the image
        - Convert to HSV
        - Count pixels with blue-ish hue and sufficient brightness
    """
    h, w, _ = img.shape
    if h == 0 or w == 0:
        return np.nan

    top_h = max(1, h // 3)
    top = img[:top_h, :, :]

    hsv = cv2.cvtColor(top, cv2.COLOR_BGR2HSV)
    hh, ss, vv = cv2.split(hsv)

    # Blue-ish hue range, moderately bright
    sky_mask = (
        (hh >= 90) & (hh <= 140) &  # blue range
        (vv >= 40)                  # not too dark
    )

    return float(sky_mask.mean())


def _compute_entropy(gray: np.ndarray) -> float:
    """
    Shannon entropy of grayscale histogram.
    """
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist = hist.ravel()
    p = hist / (hist.sum() + 1e-8)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def compute_image_features(img_path: str) -> dict:
    """
    Compute all Phase 2 engineered features for a single image.

    Returns a dict with keys:
        laplacian_var, dark_channel_prior, rms_contrast,
        sat_mean, sat_std, edge_density, sky_fraction,
        entropy, bright_skew, bright_kurtosis,
        rb_ratio, gb_ratio
    """
    img = cv2.imread(img_path)
    if img is None:
        return {
            "laplacian_var": np.nan,
            "dark_channel_prior": np.nan,
            "rms_contrast": np.nan,
            "sat_mean": np.nan,
            "sat_std": np.nan,
            "edge_density": np.nan,
            "sky_fraction": np.nan,
            "entropy": np.nan,
            "bright_skew": np.nan,
            "bright_kurtosis": np.nan,
            "rb_ratio": np.nan,
            "gb_ratio": np.nan,
        }

    # Optionally resize for speed/consistency
    target_size = 256
    h, w = img.shape[:2]
    if max(h, w) > target_size:
        scale = target_size / max(h, w)
        img = cv2.resize(
            img,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_AREA,
        )

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Laplacian variance (sharpness proxy)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # RMS contrast (std dev of gray)
    rms_contrast = float(gray.std())

    # Saturation mean/std
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    _, s, _ = cv2.split(hsv)
    sat_mean = float(s.mean())
    sat_std = float(s.std())

    # Edge density
    edges = cv2.Canny(gray, 50, 150)
    edge_density = float(np.count_nonzero(edges) / edges.size)

    # Sky fraction (heuristic)
    sky_fraction = _compute_sky_fraction(img)

    # Entropy of brightness
    entropy = _compute_entropy(gray)

    # Brightness skew/kurtosis
    g = gray.astype(np.float32).ravel()
    mu = float(g.mean())
    sigma = float(g.std() + 1e-8)
    norm = (g - mu) / sigma
    bright_skew = float((norm ** 3).mean())
    bright_kurtosis = float((norm ** 4).mean() - 3.0)  # excess kurtosis

    # Color ratios
    b, g_ch, r = cv2.split(img)
    mean_r = float(r.mean())
    mean_g = float(g_ch.mean())
    mean_b = float(b.mean() + 1e-6)

    rb_ratio = float(mean_r / mean_b)
    gb_ratio = float(mean_g / mean_b)

    # Dark channel prior
    dcp = _compute_dark_channel_prior(img)

    return {
        "laplacian_var": lap_var,
        "dark_channel_prior": dcp,
        "rms_contrast": rms_contrast,
        "sat_mean": sat_mean,
        "sat_std": sat_std,
        "edge_density": edge_density,
        "sky_fraction": sky_fraction,
        "entropy": entropy,
        "bright_skew": bright_skew,
        "bright_kurtosis": bright_kurtosis,
        "rb_ratio": rb_ratio,
        "gb_ratio": gb_ratio,
    }


# ----------------------------------------------------
#  PHASE 2 MAIN PIPELINE
# ----------------------------------------------------


FEATURE_COLS = [
    "laplacian_var",
    "dark_channel_prior",
    "rms_contrast",
    "sat_mean",
    "sat_std",
    "edge_density",
    "sky_fraction",
    "entropy",
    "bright_skew",
    "bright_kurtosis",
    "rb_ratio",
    "gb_ratio",
]


def _apply_features(df: pd.DataFrame) -> pd.DataFrame:
    feats = []
    for p in df["image_path"]:
        feats.append(compute_image_features(p))
    feat_df = pd.DataFrame(feats)
    return pd.concat([df.reset_index(drop=True), feat_df], axis=1)


def _compute_zscore_stats(df_ind_nep: pd.DataFrame) -> dict:
    """
    Compute mean/std for each feature using IND_NEP dataset stats
    (i.e., metadata_image_only.csv rows).
    """
    stats: dict[str, dict[str, float]] = {}
    for col in FEATURE_COLS:
        vals = df_ind_nep[col].to_numpy(dtype=float)
        mask = np.isfinite(vals)
        if not mask.any():
            mean = 0.0
            std = 1.0
        else:
            mean = float(vals[mask].mean())
            std = float(vals[mask].std() or 1.0)
        stats[col] = {"mean": mean, "std": std}
    return stats


def _apply_zscore(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    for col in FEATURE_COLS:
        mean = stats[col]["mean"]
        std = stats[col]["std"]
        z_col = f"{col}_z"
        df[z_col] = (df[col].astype(float) - mean) / std
    return df


def run_phase2():
    # Load Phase 1 outputs
    df_img = pd.read_csv("metadata_image_only.csv")
    df_fus = pd.read_csv("metadata_fusion.csv")

    # 1) Compute raw features
    df_img2 = _apply_features(df_img)
    df_fus2 = _apply_features(df_fus)

    # 2) Compute Z-score stats using IND_NEP (image-only metadata)
    stats = _compute_zscore_stats(df_img2)

    # 3) Apply Z-score normalization to both image-only and fusion
    df_img2 = _apply_zscore(df_img2, stats)
    df_fus2 = _apply_zscore(df_fus2, stats)

    # 4) Save feature stats
    with Path("feature_stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    # 5) Save enriched CSVs
    df_img2.to_csv("features_image_only.csv", index=False)
    df_fus2.to_csv("features_fusion.csv", index=False)

    return df_img2, df_fus2, stats


# ----------------------------------------------------
#  ENTRY POINT
# ----------------------------------------------------


if __name__ == "__main__":
    device = init_env()

    df_img2, df_fus2, stats = run_phase2()

    log_run(
        phase="phase2_engineered_features",
        stage="full_features_v1",
        description="Phase 2 engineered features + Z-score computed.",
        config={"device": str(device)},
        metrics={
            "rows_image_only": len(df_img2),
            "rows_fusion": len(df_fus2),
            "num_features_raw": len(FEATURE_COLS),
        },
    )

    print("Phase 2 complete.")

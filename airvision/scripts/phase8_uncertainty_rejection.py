#!/usr/bin/env python3
"""
PHASE 8 — UNCERTAINTY & SELECTIVE PREDICTION (FINAL)

Uncertainty:
- Epistemic only (MC Dropout)
- Aleatoric (aqi_logvar) NOT used for rejection

No training. No optimizer. No gradients.
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from src.models.phase4_model_old import AQIModel, AQIModelConfig
from src.training.phase5_stage3 import Stage3Config, load_phase2_embeddings
from src.utils.env_utils import init_env

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
MC_PASSES = 20
SIGMAS = [float("inf"), 100, 80, 60, 50, 40, 30, 20]

CKPT_PATH = Path("checkpoints/phase6_B4_fold0.pth")
VAL_SPLIT = Path("splits/fold0_val.csv")

OUT_DIR = Path("results/phase8")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# ENV
# --------------------------------------------------
device = init_env(anomaly_for_debug=False)
stage3_cfg = Stage3Config()

# --------------------------------------------------
# LOAD CHECKPOINT
# --------------------------------------------------
ckpt = torch.load(CKPT_PATH, map_location=device)

# --------------------------------------------------
# REBUILD MODEL — EXACT PHASE-6 B4
# --------------------------------------------------
model_cfg = AQIModelConfig(
    img_backbone_out_dim=1280,
    engineered_dim=524,
    engineered_proj_dim=64,
    weather_dim=0,        # Phase-7 skipped
    haze_num_classes=6,
    dropout_p=0.3,
    use_imagenet_weights=True,
)

model = AQIModel(model_cfg).to(device)
model.load_state_dict(ckpt["model_state"], strict=False)

model.train()                 # enable dropout
torch.set_grad_enabled(False) # no gradients

# --------------------------------------------------
# LOAD PHASE-2 EMBEDDINGS
# --------------------------------------------------
emb_map = load_phase2_embeddings(stage3_cfg)

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
df = pd.read_csv(VAL_SPLIT)

required = [
    "image_id",
    "image_path",
    "AQI",
    "Hour",
    "conf_twilight",
    "is_low_information",
]
missing = [c for c in required if c not in df.columns]
if missing:
    raise RuntimeError(f"Missing required columns: {missing}")

# --------------------------------------------------
# DERIVE SUN STATUS (CORRECT)
# --------------------------------------------------
df["hour_int"] = df["Hour"].str.split(":").str[0].astype(int)

df["sun_status"] = np.where(
    (df["hour_int"] < 6) |
    (df["hour_int"] >= 18) |
    (df["conf_twilight"] == 1),
    "night",
    "day",
)

# --------------------------------------------------
# IMAGE TRANSFORMS
# --------------------------------------------------
from torchvision.io import read_image
from torchvision.transforms import v2 as T

transform = T.Compose([
    T.ConvertImageDtype(torch.float32),
    T.Resize(256),
    T.CenterCrop(224),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

# --------------------------------------------------
# MC DROPOUT INFERENCE
# --------------------------------------------------
records = []

for _, r in tqdm(df.iterrows(), total=len(df), desc="MC Dropout"):
    image_id = int(r["image_id"])

    img = read_image(r["image_path"])
    img = transform(img).unsqueeze(0).to(device)

    if image_id in emb_map:
        emb = torch.from_numpy(emb_map[image_id]).float().unsqueeze(0).to(device)
    else:
        emb = torch.zeros((1, 524), device=device)

    preds = []
    for _ in range(MC_PASSES):
        out = model(img, emb)
        preds.append(out["aqi_mean"].item())

    preds = np.asarray(preds)

    records.append({
        "image_id": image_id,
        "aqi_true": float(r["AQI"]),
        "aqi_pred_mean": float(preds.mean()),
        "aqi_pred_std": float(preds.std()),
        "sun_status": r["sun_status"],
        "is_low_information": bool(r["is_low_information"]),
    })

df_pred = pd.DataFrame(records)
df_pred.to_csv(OUT_DIR / "phase8_predictions.csv", index=False)

# --------------------------------------------------
# REJECTION + METRICS
# --------------------------------------------------
rows = []

for sigma in SIGMAS:
    rejected = np.zeros(len(df_pred), dtype=bool)

    rejected |= (df_pred["sun_status"] == "night")
    rejected |= df_pred["is_low_information"]

    if np.isfinite(sigma):
        rejected |= df_pred["aqi_pred_std"] > sigma

    accepted = ~rejected
    coverage = accepted.mean()

    if accepted.sum() == 0:
        mae = rmse = np.nan
    else:
        err = df_pred.loc[accepted, "aqi_pred_mean"] - df_pred.loc[accepted, "aqi_true"]
        mae  = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err ** 2)))

    rows.append({
        "sigma": sigma,
        "coverage": coverage,
        "mae": mae,
        "rmse": rmse,
    })

pd.DataFrame(rows).to_csv(OUT_DIR / "coverage_vs_metrics.csv", index=False)

print("✅ PHASE-8 COMPLETE")
print("Saved:")
print(" - results/phase8/phase8_predictions.csv")
print(" - results/phase8/coverage_vs_metrics.csv")

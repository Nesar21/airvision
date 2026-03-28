from __future__ import annotations

import pathlib

# Root paths
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]

DATA_ROOT = PROJECT_ROOT / "kaggle"
PM25VISION_ROOT = PROJECT_ROOT / "pm25vision"

# IMPORTANT: these datasets are intentionally ignored in MASTER PLAN v2.5
IGNORED_DATASETS = {
    "SAPID": DATA_ROOT / "Smartphone-Based Air Pollution Image Dataset (SAPID)",
    "LIME_Explanation": DATA_ROOT / "LIME_Explanation",
}

# Environment + training rules
GLOBAL_SEED: int = 42
USE_FLOAT32_ONLY: bool = True
GRAD_CLIP_MAX_NORM: float = 1.0

RUN_MANIFEST_PATH = PROJECT_ROOT / "run_manifest.json"

# catchphrase / experiment family (you can change string later)
EXPERIMENT_FAMILY = "AQI_MASTER_V2_5"

def print_config_summary() -> None:
    print("=== MASTER PLAN v2.5 CONFIG ===")
    print(f"PROJECT_ROOT        : {PROJECT_ROOT}")
    print(f"DATA_ROOT           : {DATA_ROOT}")
    print(f"PM25VISION_ROOT     : {PM25VISION_ROOT}")
    print(f"RUN_MANIFEST_PATH   : {RUN_MANIFEST_PATH}")
    print(f"GLOBAL_SEED         : {GLOBAL_SEED}")
    print(f"USE_FLOAT32_ONLY    : {USE_FLOAT32_ONLY}")
    print(f"GRAD_CLIP_MAX_NORM  : {GRAD_CLIP_MAX_NORM}")
    print(f"EXPERIMENT_FAMILY   : {EXPERIMENT_FAMILY}")
    print("IGNORING DATASETS   :")
    for name, path in IGNORED_DATASETS.items():
        print(f"  - {name}: {path}")

if __name__ == "__main__":
    print_config_summary()

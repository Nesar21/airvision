from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
import json

from src.utils.env_utils import init_env
from src.utils.manifest import log_run


# ------------------------------------------------------------
# Normalize engineered features using stats.json
# ------------------------------------------------------------
def normalize_engineered(feats: pd.DataFrame, stats: dict) -> np.ndarray:
    """
    Z-score normalize engineered features.
    Column order is determined by stats_json keys (stable, reproducible).
    """
    cols = list(stats.keys())  # ordered list of engineered feature names

    out = []
    for c in cols:
        mean = stats[c]["mean"]
        std = stats[c]["std"]
        if std < 1e-6:
            std = 1.0
        out.append((feats[c].values - mean) / std)

    return np.vstack(out).T.astype(np.float32)   # [N, num_features]


# ------------------------------------------------------------
# Finalize a dataset split (image-only or fusion)
# ------------------------------------------------------------
def finalize_set(
    meta_csv: str,
    feat_csv: str,
    embedding_npy: str,
    stats_json: dict,
    out_feature_npy: str,
    out_index_csv: str,
):
    # 1) Load metadata (kept for alignment)
    meta = pd.read_csv(meta_csv)

    # 2) Load engineered features CSV
    feats = pd.read_csv(feat_csv)

    # keep ONLY engineered feature columns (12 features)
    engineered_cols = list(stats_json.keys())
    feats = feats[engineered_cols]

    # 3) Load embeddings
    emb = np.load(embedding_npy)
    assert emb.shape[0] == feats.shape[0], \
        f"Embedding count {emb.shape[0]} != feature rows {feats.shape[0]}"

    # 4) Normalize engineered features
    feats_z = normalize_engineered(feats, stats_json)

    # 5) Concatenate engineered features + embeddings
    final = np.concatenate(
        [feats_z.astype(np.float32), emb.astype(np.float32)],
        axis=1
    )

    # 6) Save outputs
    np.save(out_feature_npy, final)
    meta.to_csv(out_index_csv, index=False)

    return final.shape


# ------------------------------------------------------------
# Driver
# ------------------------------------------------------------
def run_phase2_finalize():
    print("[Phase2B] Loading inputs...")

    # Load normalization stats
    with open("feature_stats.json") as f:
        stats = json.load(f)

    # IMAGE-ONLY
    shape_img = finalize_set(
        meta_csv="metadata_image_only.csv",
        feat_csv="features_image_only.csv",
        embedding_npy="clip_embeddings_image_only.npy",
        stats_json=stats,
        out_feature_npy="phase2_features_final_image_only.npy",
        out_index_csv="phase2_index_image_only.csv",
    )
    print("[Phase2B] Image-only matrix:", shape_img)

    # FUSION
    shape_fusion = finalize_set(
        meta_csv="metadata_fusion.csv",
        feat_csv="features_fusion.csv",
        embedding_npy="clip_embeddings_fusion.npy",
        stats_json=stats,
        out_feature_npy="phase2_features_final_fusion.npy",
        out_index_csv="phase2_index_fusion.csv",
    )
    print("[Phase2B] Fusion matrix:", shape_fusion)

    log_run(
        phase="phase2_finalize",
        stage="final_matrix",
        description="Final engineered + CLIP feature matrices created.",
        config={"engineered_dim": len(stats)},
        metrics={"samples_image_only": shape_img[0], "samples_fusion": shape_fusion[0]},
    )

    print("Phase 2B complete.")


if __name__ == "__main__":
    init_env()
    run_phase2_finalize()

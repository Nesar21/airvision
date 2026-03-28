# src/features/phase2_embeddings.py
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
import torch
from tqdm import tqdm

from src.utils.env_utils import init_env
from src.utils.manifest import log_run

import clip

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
MODEL_NAME = "ViT-B/32"       # CLIP ViT-B/32
EMBED_DIM = 512               # Output dimension for ViT-B/32
BATCH_SIZE = 16


def load_clip_model(device):
    model, preprocess = clip.load(MODEL_NAME, device=device, jit=False)
    return model, preprocess


def compute_embeddings(df: pd.DataFrame, model, preprocess, device):
    paths = df["image_path"].tolist()
    all_embeds = []

    for i in tqdm(range(0, len(paths), BATCH_SIZE)):
        batch = paths[i:i+BATCH_SIZE]
        images = []

        for p in batch:
            try:
                img = preprocess(clip.load_image(p)).unsqueeze(0)
                images.append(img)
            except:
                # fallback: zero embedding if corrupted
                images.append(None)

        valid_imgs = [img for img in images if img is not None]

        if len(valid_imgs) == 0:
            # all corrupted
            zeros = np.zeros((len(batch), EMBED_DIM), dtype=np.float32)
            all_embeds.append(zeros)
            continue

        imgs_tensor = torch.cat(valid_imgs).to(device)

        with torch.no_grad():
            feats = model.encode_image(imgs_tensor)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            feats = feats.float().cpu().numpy()

        # align to batch size (if some images were None → fill zeros)
        final_batch = []
        j = 0
        for img in images:
            if img is None:
                final_batch.append(np.zeros((EMBED_DIM,), dtype=np.float32))
            else:
                final_batch.append(feats[j])
                j += 1

        all_embeds.append(np.vstack(final_batch))

    return np.vstack(all_embeds)


def run_phase2_embeddings():
    device = init_env()
    model, preprocess = load_clip_model(device)

    # Load metadata
    df_img = pd.read_csv("metadata_image_only.csv")
    df_fus = pd.read_csv("metadata_fusion.csv")

    # Compute embeddings
    print("[Phase2A] Computing embeddings for image-only set...")
    emb_img = compute_embeddings(df_img, model, preprocess, device)
    np.save("clip_embeddings_image_only.npy", emb_img)

    print("[Phase2A] Computing embeddings for fusion set...")
    emb_fus = compute_embeddings(df_fus, model, preprocess, device)
    np.save("clip_embeddings_fusion.npy", emb_fus)

    log_run(
        phase="phase2_embeddings",
        stage="clip_vit",
        description="CLIP/VIT embeddings generated.",
        config={"model": MODEL_NAME},
        metrics={"image_only_rows": len(df_img), "fusion_rows": len(df_fus)},
    )

    print("Phase 2A complete.")
    print("Saved: clip_embeddings_image_only.npy, clip_embeddings_fusion.npy")


if __name__ == "__main__":
    run_phase2_embeddings()

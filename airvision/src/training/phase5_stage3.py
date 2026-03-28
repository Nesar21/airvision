#!/usr/bin/env python3
"""
Phase 5 — Stage 3: IND_NEP Fine-Tuning (5-Fold CV)
Master Plan v2.5 — FINAL IMPLEMENTATION

This rewrite uses:
    - 524D Phase-2 embeddings:
          phase2_features_final_image_only.npy
          phase2_index_image_only.csv

Official Stage-3 implementation:
    - Loads backbone from Stage 2 (or Stage 1 fallback)
    - Trains new 524D head with heteroscedastic AQI + haze aux
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import Dict, List

import math
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from src.utils.env_utils import init_env
from src.utils.manifest import log_run
from src.models.phase4_model import Phase4Model
from src.config import GRAD_CLIP_MAX_NORM


# ---------------------------------------------------------
# CONFIG — MASTER PLAN v2.5
# ---------------------------------------------------------


class Stage3Config:
    project_root = Path(".")

    # -------------------------
    # Phase-6 ablation toggles
    # -------------------------
    use_phase2_embedding = bool(int(os.getenv("USE_PHASE2", "1")))
    use_haze_head = bool(int(os.getenv("USE_HAZE", "1")))
    use_vis_head = bool(int(os.getenv("USE_VIS", "0")))
    use_uncertainty = bool(int(os.getenv("USE_UNC", "1")))

    # -------------------------
    # Phase-2 embeddings + index
    # -------------------------
    phase2_emb = project_root / "phase2_features_final_image_only.npy"
    phase2_idx = project_root / "phase2_index_image_only.csv"

    # IND_NEP metadata (not strictly needed here, but kept for completeness)
    metadata_csv = project_root / "metadata_image_only.csv"

    # -------------------------
    # Splits + checkpoints
    # -------------------------
    splits_root = project_root / "splits"
    checkpoints_root = project_root / "checkpoints"

    num_folds = 5
    aqi_bins = [0, 50, 100, 150, 200, 300, 10000]

    # -------------------------
    # Training hyperparameters
    # -------------------------
    batch_size = 16
    num_workers = 0
    max_epochs = 32        # ignored in Phase-6 (hard cap = 10)
    early_stop_patience = 8

    # -------------------------
    # Optimizer
    # -------------------------
    backbone_lr = 1e-5
    head_lr = 1e-3
    weight_decay = 1e-5
    grad_clip = GRAD_CLIP_MAX_NORM

    # -------------------------
    # Loss weights
    # -------------------------
    lambda_aqi = 1.0
    lambda_haze = 0.1
    lambda_vis = 0.0  # visibility unused in loss for now

    # -------------------------
    # Checkpoints
    # -------------------------
    ckpt_stage1 = checkpoints_root / "backbone_haze.pth"
    ckpt_stage2 = checkpoints_root / "backbone_global.pth"

    def final_ckpt_for_fold(self, fold: int) -> Path:
        return self.checkpoints_root / f"aqi_final_fold{fold}.pth"


# ---------------------------------------------------------
# UTILITIES
# ---------------------------------------------------------


AQI_BINS = [0, 50, 100, 150, 200, 300, 10000]


def bin_aqi_scalar(aqi: float) -> int:
    return int(np.digitize([aqi], AQI_BINS, right=False)[0])


def detect_column(df: pd.DataFrame, base: str) -> str:
    """
    Robust column resolver:
    - For current cleaned splits, this will just return the base column
      (AQI, AQI_Class, image_path, label_confidence).
    - Kept generic in case of accidental suffixes.
    """
    if base in df.columns:
        return base

    candidates = [c for c in df.columns if c.startswith(base + "_")]
    if not candidates:
        raise KeyError(f"Column {base} not found.")

    # SPECIAL CASE: AQI → numeric selection
    if base == "AQI":
        numeric = [c for c in candidates if pd.api.types.is_numeric_dtype(df[c])]
        if numeric:
            return numeric[0]
        return candidates[0]

    # SPECIAL CASE: AQI_Class → string selection
    if base == "AQI_Class":
        stringy = [c for c in candidates if df[c].dtype == object]
        if stringy:
            return stringy[0]
        return candidates[0]

    # General rule: pick *_x before *_y
    candidates.sort()
    return candidates[0]


def build_haze_label_mapping() -> Dict[str, int]:
    """Maps AQI_Class to haze class index."""
    return {
        "a_Good": 0,
        "b_Moderate": 1,
        "c_Unhealthy_for_Sensitive_Groups": 2,
        "d_Unhealthy": 3,
        "e_Very_Unhealthy": 4,
        "f_Severe": 5,
    }


# ---------------------------------------------------------
# LOAD PHASE-2 EMBEDDINGS + MAP BY image_id
# ---------------------------------------------------------


def load_phase2_embeddings(cfg: Stage3Config) -> Dict[int, np.ndarray]:
    emb = np.load(cfg.phase2_emb)              # [N, 524]
    df_idx = pd.read_csv(cfg.phase2_idx)       # contains image_id column

    if "image_id" not in df_idx.columns:
        raise ValueError("phase2_index_image_only.csv must contain image_id")

    mapping: Dict[int, np.ndarray] = {}
    for i, row in df_idx.iterrows():
        image_id = int(row["image_id"])
        mapping[image_id] = emb[i]             # 524D vector

    return mapping


# ---------------------------------------------------------
# DATASET (IND-NEP)
# ---------------------------------------------------------


class INDNEPAQIDataset(Dataset):
    """
    Stage 3 dataset using:

        - image
        - 524D embedding from Phase 2
        - true AQI
        - haze label
        - label confidence
        - city (for metrics)
        - AQI bin (for sampler)
    """

    def __init__(
        self,
        fold_csv: Path,
        emb_map: Dict[int, np.ndarray],
        device: torch.device,
    ) -> None:
        from torchvision.io import read_image
        from torchvision.transforms import v2 as T

        self.df = pd.read_csv(fold_csv)
        self.emb_map = emb_map
        self.device = device

        self.read_image = read_image
        self.transform = T.Compose(
            [
                T.ConvertImageDtype(torch.float32),
                T.Resize(256),
                T.CenterCrop(224),
                T.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        self.col_image_path = detect_column(self.df, "image_path")
        self.col_aqi = detect_column(self.df, "AQI")
        self.col_aqi_class = detect_column(self.df, "AQI_Class")
        self.col_label_conf = detect_column(self.df, "label_confidence")

        if "image_id" not in self.df.columns:
            raise KeyError("'image_id' must exist in the fold CSV.")
        if "city" not in self.df.columns:
            raise KeyError("'city' must exist in fold CSV.")

        self.haze_map = build_haze_label_mapping()

        # Prebuild rows
        self.rows: List[Dict] = []
        for _, r in self.df.iterrows():
            imid = int(r["image_id"])
            aqi = float(r[self.col_aqi])
            aqi_class = str(r[self.col_aqi_class])
            label_conf = float(r[self.col_label_conf])
            city = str(r["city"])

            haze_label = self.haze_map.get(aqi_class, 1)  # fallback Moderate
            aqi_bin = bin_aqi_scalar(aqi)

            self.rows.append(
                {
                    "image_id": imid,
                    "image_path": r[self.col_image_path],
                    "aqi": aqi,
                    "aqi_norm": aqi / 500.0,
                    "label_conf": label_conf,
                    "haze_label": haze_label,
                    "city": city,
                    "aqi_bin": aqi_bin,
                }
            )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        r = self.rows[idx]

        # Image
        img = self.read_image(str(r["image_path"]))
        img = self.transform(img)

        # 524D embedding
        imid = r["image_id"]
        if imid not in self.emb_map:
            emb = torch.zeros(524, dtype=torch.float32)
        else:
            emb = torch.from_numpy(self.emb_map[imid]).float()

        return {
            "images": img,
            "embedding": emb,
            "aqi": torch.tensor(r["aqi"], dtype=torch.float32),
            "aqi_norm": torch.tensor(r["aqi_norm"], dtype=torch.float32),
            "label_conf": torch.tensor(r["label_conf"], dtype=torch.float32),
            "haze_label": torch.tensor(r["haze_label"], dtype=torch.long),
            "city": r["city"],
            "aqi_bin": r["aqi_bin"],
        }


# ---------------------------------------------------------
# SAMPLER — LONG-TAIL CORRECTION
# ---------------------------------------------------------


def build_long_tail_sampler(df: pd.DataFrame, col_aqi: str) -> WeightedRandomSampler:
    """
    WeightedRandomSampler with:
        w_bin = 1 / freq_bin
        if AQI > 250: w = w_bin * 10
        else:         w = w_bin
    """
    aqi_values = df[col_aqi].astype(float).values
    bins = np.digitize(aqi_values, AQI_BINS, right=False)

    freq: Dict[int, int] = {}
    for b in bins:
        freq[b] = freq.get(b, 0) + 1

    weights = []
    for aqi, b in zip(aqi_values, bins):
        w_bin = 1.0 / float(freq[b])
        if aqi > 250.0:
            w = w_bin * 10.0
        else:
            w = w_bin
        weights.append(w)

    weights_tensor = torch.tensor(weights, dtype=torch.float32)
    return WeightedRandomSampler(
        weights=weights_tensor,
        num_samples=len(weights),
        replacement=True,
    )


# ---------------------------------------------------------
# MODEL + OPTIM — BACKBONE-ONLY LOADING
# ---------------------------------------------------------


def build_model_and_optim(cfg: Stage3Config, device: torch.device):
    """
    Phase-6 ablation-ready model builder.

    - Uses Phase4Model with ON/OFF toggles
    - Loads ONLY backbone weights from Stage-2 / Stage-1
    - Heads are freshly initialized every run
    - Supports A1 → B4 without code changes
    """

    # -------------------------------------------------
    # Build ablation-capable model (TOGGLES DRIVE IT)
    # -------------------------------------------------
    model = Phase4Model(
        use_phase2_embedding=cfg.use_phase2_embedding,
        use_haze_head=cfg.use_haze_head,
        use_vis_head=cfg.use_vis_head,
        use_uncertainty=cfg.use_uncertainty,
        pretrained_backbone=False,
    ).to(device)

    # -------------------------------------------------
    # Load backbone-only weights (NO heads)
    # -------------------------------------------------
    ckpt_path = cfg.ckpt_stage2 if cfg.ckpt_stage2.exists() else cfg.ckpt_stage1
    ckpt = torch.load(ckpt_path, map_location=device)

    state = ckpt.get("model_state", ckpt)

    # Keep ONLY backbone.* params
    backbone_state = {
        k.replace("backbone.", ""): v
        for k, v in state.items()
        if k.startswith("backbone.")
    }

    missing, unexpected = model.backbone.load_state_dict(
        backbone_state, strict=False
    )

    print(f"[Stage3] Loaded backbone from {ckpt_path}")
    print(f"[Stage3] backbone missing keys: {missing}")
    print(f"[Stage3] backbone unexpected keys: {unexpected}")

    # -------------------------------------------------
    # Optimizer (backbone LR << head LR)
    # -------------------------------------------------
    backbone_params = list(model.backbone.parameters())

    head_params = []
    if model.engineered_proj is not None:
        head_params += list(model.engineered_proj.parameters())
    head_params += list(model.aqi_head.parameters())

    if model.logvar_head is not None:
        head_params += list(model.logvar_head.parameters())
    if model.haze_head is not None:
        head_params += list(model.haze_head.parameters())
    if model.vis_head is not None:
        head_params += list(model.vis_head.parameters())

    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": cfg.backbone_lr},
            {"params": head_params, "lr": cfg.head_lr},
        ],
        weight_decay=cfg.weight_decay,
    )


    # -------------------------------------------------
    # Minimal model summary for logs
    # -------------------------------------------------
    model_cfg = {
        "use_phase2_embedding": cfg.use_phase2_embedding,
        "use_haze_head": cfg.use_haze_head,
        "use_vis_head": cfg.use_vis_head,
        "use_uncertainty": cfg.use_uncertainty,
        "combined_dim": model.combined_dim,
    }

    return model, optimizer, model_cfg

# ---------------------------------------------------------
# LOSSES
# ---------------------------------------------------------


def hetero_aqi_loss_per_sample(
    mean_norm: torch.Tensor,
    logvar: torch.Tensor,
    target_norm: torch.Tensor,
) -> torch.Tensor:
    """
    Per-sample heteroscedastic NLL on normalized AQI:
        mean_norm, logvar, target_norm: [B]
        returns [B]
    """
    logvar = torch.clamp(logvar, -10.0, 10.0)
    var_inv = torch.exp(-logvar)
    sq = (target_norm - mean_norm) ** 2
    nll = 0.5 * (var_inv * sq + logvar)
    return nll


# ---------------------------------------------------------
# METRICS
# ---------------------------------------------------------


def compute_epoch_metrics(
    preds_aqi: List[float],
    targets_aqi: List[float],
    cities: List[str],
) -> Dict[str, float]:
    y_pred = np.array(preds_aqi, dtype=np.float32)
    y_true = np.array(targets_aqi, dtype=np.float32)
    assert y_pred.shape == y_true.shape

    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    var = float(np.var(y_true))
    if var < 1e-8:
        r2 = 0.0
    else:
        r2 = float(1.0 - np.sum(err ** 2) / (np.sum((y_true - y_true.mean()) ** 2) + 1e-8))

    within25 = float(np.mean(np.abs(err) <= 25.0))
    within50 = float(np.mean(np.abs(err) <= 50.0))

    # per-bin MAE
    bin_mae: Dict[str, float] = {}
    for b_idx in range(len(AQI_BINS) - 1):
        low = AQI_BINS[b_idx]
        high = AQI_BINS[b_idx + 1]
        mask = (y_true >= low) & (y_true < high)
        if mask.any():
            bin_mae[f"bin_{low}_{high}"] = float(np.mean(np.abs(err[mask])))

    # per-city MAE
    city_mae: Dict[str, float] = {}
    city_groups: Dict[str, List[float]] = defaultdict(list)
    for e, c in zip(err, cities):
        city_groups[c].append(float(e))
    for c, errs in city_groups.items():
        city_mae[f"city_{c}"] = float(np.mean(np.abs(np.array(errs))))

    metrics = {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "within25": within25,
        "within50": within50,
    }
    metrics.update(bin_mae)
    metrics.update(city_mae)
    return metrics


# ---------------------------------------------------------
# EPOCH LOOP
# ---------------------------------------------------------


def run_epoch(
    model: Phase4Model,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    cfg: Stage3Config,
    train: bool = True,
) -> Dict[str, float]:
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_samples = 0

    all_preds_aqi: List[float] = []
    all_targets_aqi: List[float] = []
    all_cities: List[str] = []

    for batch in loader:
        images = batch["images"].to(device)
        emb = batch["embedding"].to(device)          # [B, 524] (ignored if disabled)
        aqi = batch["aqi"].to(device)                # raw AQI
        aqi_norm = batch["aqi_norm"].to(device)      # AQI / 500
        label_conf = batch["label_conf"].to(device)  # [B]
        haze_label = batch["haze_label"].to(device)  # [B]

        B = images.size(0)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            out = model(images, emb)

            # ---------------------------------
            # AQI mean (ALWAYS present)
            # ---------------------------------
            mean_raw = out["aqi_mean"].squeeze(-1)   # [B]
            mean_norm = mean_raw / 500.0

            # ---------------------------------
            # Uncertainty (OPTIONAL)
            # ---------------------------------
            if out.get("aqi_logvar") is not None:
                logvar = out["aqi_logvar"].squeeze(-1)
            else:
                logvar = torch.zeros_like(mean_norm)

            nll_aqi = hetero_aqi_loss_per_sample(
                mean_norm, logvar, aqi_norm
            )  # [B]

            # ---------------------------------
            # Haze (OPTIONAL)
            # ---------------------------------
            if out.get("haze_logits") is not None:
                haze_logits = out["haze_logits"]
                ce_haze = F.cross_entropy(
                    haze_logits,
                    haze_label,
                    reduction="none",
                )
            else:
                ce_haze = torch.zeros_like(nll_aqi)

            # ---------------------------------
            # Final weighted loss
            # ---------------------------------
            per_sample_loss = label_conf * (
                cfg.lambda_aqi * nll_aqi +
                cfg.lambda_haze * ce_haze
            )

            loss = per_sample_loss.mean()

            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.grad_clip
                )
                optimizer.step()

        total_loss += loss.item() * B
        total_samples += B

        all_preds_aqi.extend(mean_raw.detach().cpu().numpy().tolist())
        all_targets_aqi.extend(aqi.detach().cpu().numpy().tolist())
        all_cities.extend(batch["city"])

    avg_loss = total_loss / max(1, total_samples)
    metrics = compute_epoch_metrics(
        all_preds_aqi, all_targets_aqi, all_cities
    )
    metrics["loss"] = avg_loss
    return metrics


# ---------------------------------------------------------
# TRAIN ONE FOLD
# ---------------------------------------------------------


def train_fold(
    fold: int,
    device: torch.device,
    cfg: Stage3Config,
    emb_map: Dict[int, np.ndarray],
):
    train_csv = cfg.splits_root / f"fold{fold}_train.csv"
    val_csv = cfg.splits_root / f"fold{fold}_val.csv"

    if not train_csv.exists() or not val_csv.exists():
        raise FileNotFoundError(
            f"Missing split CSV for fold {fold}: {train_csv} / {val_csv}"
        )

    df_train = pd.read_csv(train_csv)
    col_aqi = detect_column(df_train, "AQI")

    sampler = build_long_tail_sampler(df_train, col_aqi)

    train_ds = INDNEPAQIDataset(train_csv, emb_map, device)
    val_ds = INDNEPAQIDataset(val_csv, emb_map, device)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=False,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=False,
    )

    model, optimizer, model_cfg = build_model_and_optim(cfg, device)

    best_val_mae = float("inf")
    best_state = None
    patience = 0

    # -------------------------------------------------
    # Phase-6 HARD CAP (fixed)
    # -------------------------------------------------
    EPOCH_CAP = 10

    for epoch in range(1, EPOCH_CAP + 1):
        train_m = run_epoch(
            model, train_loader, optimizer, device, cfg, train=True
        )
        val_m = run_epoch(
            model, val_loader, optimizer, device, cfg, train=False
        )

        print(
            f"[Stage3][Fold {fold}][Epoch {epoch:03d}] "
            f"train_loss={train_m['loss']:.4f} train_mae={train_m['mae']:.2f} "
            f"val_loss={val_m['loss']:.4f} val_mae={val_m['mae']:.2f}"
        )

        if val_m["mae"] < best_val_mae:
            best_val_mae = val_m["mae"]
            best_state = {
                "model_state": model.state_dict(),
                "model_cfg": model_cfg,   # ✅ FIXED
                "epoch": epoch,
                "val_metrics": val_m,
            }
            patience = 0
        else:
            patience += 1
            if patience >= cfg.early_stop_patience:
                print(
                    f"[Stage3][Fold {fold}] Early stopping at epoch {epoch}"
                )
                break

    if best_state is None:
        print(f"[Stage3][Fold {fold}] WARNING: No best_state saved.")
        return

    ckpt_path = cfg.final_ckpt_for_fold(fold)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, ckpt_path)

    print(
        f"[Stage3][Fold {fold}] Saved final checkpoint → {ckpt_path}"
    )
    print(
        f"[Stage3][Fold {fold}] Best val MAE = {best_val_mae:.2f}"
    )

    log_run(
        phase="phase6_ablation",
        stage=f"ablation_fold{fold}",
        description="Phase-6 ablation (Fold-0, 10 epochs, component toggles only).",
        config={
            "fold": fold,
            "epoch_cap": EPOCH_CAP,
            "batch_size": cfg.batch_size,
            "backbone_lr": cfg.backbone_lr,
            "head_lr": cfg.head_lr,
            "lambda_aqi": cfg.lambda_aqi,
            "lambda_haze": cfg.lambda_haze,
        },
        metrics=best_state["val_metrics"],
    )

# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------


def main():
    device = init_env(anomaly_for_debug=False)
    cfg = Stage3Config()

    emb_map = load_phase2_embeddings(cfg)
    print(f"[Stage3] Loaded Phase-2 embeddings for {len(emb_map)} images.")

    # Phase-6 guard: ONLY run Fold-0
    for fold in [0]:
        print(f"\n========== Stage 3 — Fold {fold} ==========")
        train_fold(fold, device, cfg, emb_map)


if __name__ == "__main__":
    main()

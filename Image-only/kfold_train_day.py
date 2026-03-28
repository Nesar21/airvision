# kfold_train_day.py
# Full-featured 4-fold trainer for DAY-ONLY dataset (Image+Depth+Numeric)
# Usage:
#   python kfold_train_day.py --config cfg_kfold_day_full.yaml

import os
import json
import time
import argparse
from pathlib import Path
from collections import defaultdict

import yaml
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from litefusion_model_and_train import (
    AQIMultiDataset,
    collate_batch,
    LiteFusion,
    masked_multi_task_loss,
    evaluate
)

# ------------------------
# Helpers
# ------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

class EMA:
    """Simple EMA wrapper for model weights."""
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.detach().cpu().clone()

    def update(self, model):
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name].mul_(self.decay)
                self.shadow[name].add_((1.0 - self.decay) * p.detach().cpu())

    def apply_to(self, model):
        self.backup = {}
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.backup[name] = p.detach().cpu().clone()
                p.data.copy_(self.shadow[name].to(p.device))

    def restore(self, model):
        for name, p in model.named_parameters():
            if p.requires_grad:
                p.data.copy_(self.backup[name].to(p.device))
        self.backup = None

# ------------------------
# Single fold training
# ------------------------
def train_one_fold(cfg, fold_id, train_csv, val_csv, test_csv, save_dir):
    device = torch.device(cfg["device"])
    os.makedirs(save_dir, exist_ok=True)

    # Dataloaders
    train_ds = AQIMultiDataset(train_csv, img_size=cfg["img_size"])
    val_ds   = AQIMultiDataset(val_csv, img_size=cfg["img_size"])
    test_ds  = AQIMultiDataset(test_csv, img_size=cfg["img_size"])

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg.get("num_workers", 4),
        collate_fn=collate_batch,
        pin_memory=cfg.get("pin_memory", False)
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg.get("num_workers", 4),
        collate_fn=collate_batch,
        pin_memory=cfg.get("pin_memory", False)
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg.get("num_workers", 4),
        collate_fn=collate_batch,
        pin_memory=cfg.get("pin_memory", False)
    )

    # Model
    model = LiteFusion(
        backbone_name=cfg["backbone"],
        pretrained=cfg["pretrained"],
        img_size=cfg["img_size"]
    ).to(device)

    # Optimizer
    if cfg.get("optimizer", "adamw").lower() == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg["lr"], momentum=cfg.get("momentum", 0.9), weight_decay=cfg["weight_decay"])
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    # Scheduler
    sched_name = cfg.get("scheduler", "cosine").lower()
    if sched_name == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=cfg["lr"], total_steps=cfg["epochs"] * max(1, len(train_loader)//max(1,cfg.get("grad_accum_steps",1))), pct_start=cfg.get("onecycle_pct_start", 0.3))
    elif sched_name == "steplr":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.get("step_size",10), gamma=cfg.get("step_gamma",0.1))
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])

    # Optional EMA
    ema = None
    if cfg.get("use_ema", False):
        ema = EMA(model, decay=cfg.get("ema_decay", 0.999))

    # Optional SWA
    use_swa = cfg.get("use_swa", False)
    swa_model = None
    swa_start = cfg.get("swa_start", int(cfg["epochs"]*0.75))
    if use_swa:
        swa_model = torch.optim.swa_utils.AveragedModel(model)

    # AMP: only enable if CUDA available (MPS AMP support is experimental)
    use_amp = cfg.get("amp", False) and torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    best_val = float("inf")
    early_patience = cfg.get("patience", 7)
    min_epochs = cfg.get("min_epochs", 8)
    grad_accum = cfg.get("grad_accum_steps", 1)

    # Logging file
    log_path = Path(save_dir) / "training_log.csv"
    if not log_path.exists():
        with open(log_path, "w") as f:
            f.write("fold,epoch,train_loss,val_mae,aqi_rmse,aqi_r2,lr,time_s\n")

    # Train loop
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        t0 = time.time()
        total_loss = 0.0
        iters = 0

        optimizer.zero_grad()
        for step, batch in enumerate(train_loader, start=1):
            img = batch["image"].to(device)
            depth = batch["depth"].to(device)

            pm25_v = torch.nan_to_num(batch["pm25"], nan=0.0).to(device)
            pm10_v = torch.nan_to_num(batch["pm10"], nan=0.0).to(device)
            aqi_bin = torch.zeros(img.shape[0], device=device)
            numeric = torch.stack([pm25_v, pm10_v, aqi_bin], dim=1)

            if use_amp:
                with torch.cuda.amp.autocast():
                    out = model(img, depth, numeric)
                    targets = {"aqi": batch["aqi"].to(device), "pm25": batch["pm25"].to(device), "pm10": batch["pm10"].to(device)}
                    masks = {"has_aqi": batch["has_aqi"].to(device), "has_pm25": batch["has_pm25"].to(device), "has_pm10": batch["has_pm10"].to(device)}
                    loss = masked_multi_task_loss(out, targets, masks, cfg) / grad_accum
                scaler.scale(loss).backward()
            else:
                out = model(img, depth, numeric)
                targets = {"aqi": batch["aqi"].to(device), "pm25": batch["pm25"].to(device), "pm10": batch["pm10"].to(device)}
                masks = {"has_aqi": batch["has_aqi"].to(device), "has_pm25": batch["has_pm25"].to(device), "has_pm10": batch["has_pm10"].to(device)}
                loss = masked_multi_task_loss(out, targets, masks, cfg) / grad_accum
                loss.backward()

            if (step % grad_accum) == 0:
                if use_amp:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.get("grad_clip", 1.0))
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.get("grad_clip", 1.0))
                    optimizer.step()
                optimizer.zero_grad()

            total_loss += (loss.item() * grad_accum) if isinstance(loss, torch.Tensor) else loss * grad_accum
            iters += 1

        # Scheduler step (per epoch)
        if sched_name != "onecycle":
            try:
                scheduler.step()
            except Exception:
                pass

        # SWA update
        if use_swa and epoch >= swa_start:
            swa_model.update_parameters(model)

        # EMA update
        if ema is not None:
            ema.update(model)

        duration = time.time() - t0

        # Validation / metrics
        # If EMA present, evaluate using EMA weights
        if ema is not None:
            ema.apply_to(model)

        val_metrics = evaluate(model, val_loader, device)

        if ema is not None:
            ema.restore(model)

        val_mae = val_metrics.get("aqi_mae", float("inf"))

        # Save best
        cur_lr = optimizer.param_groups[0]["lr"]
        print(f"[Fold {fold_id}] Epoch {epoch}/{cfg['epochs']} | train_loss={total_loss/iters:.4f} | val_mae={val_mae:.4f} | lr={cur_lr:.5g} | time={duration:.1f}s")
        with open(log_path, "a") as f:
            f.write(",".join(map(str, [fold_id, epoch, total_loss/iters, val_metrics.get("aqi_mae", None), val_metrics.get("aqi_rmse", None), val_metrics.get("aqi_r2", None), cur_lr, round(duration,1)])) + "\n")

        if val_mae < best_val:
            best_val = val_mae
            torch.save(model.state_dict(), Path(save_dir) / "best.pth")
            # also save optimizer state
            torch.save(optimizer.state_dict(), Path(save_dir) / "optim.pth")
            patience_cnt = 0
        else:
            patience_cnt = locals().get("patience_cnt", 0) + 1

        # Early stopping check
        if epoch >= min_epochs and patience_cnt > early_patience:
            print(f"[Fold {fold_id}] Early stopping at epoch {epoch} (patience={early_patience})")
            break

    # Finalize SWA (if used)
    if use_swa:
        torch.optim.swa_utils.update_bn(train_loader, swa_model)
        torch.save(swa_model.module.state_dict(), Path(save_dir) / "swa.pth")

    # Test evaluation using best.pth (or swa if configured to use it)
    final_ckpt = Path(save_dir) / ("swa.pth" if use_swa and (Path(save_dir)/"swa.pth").exists() else "best.pth")
    model.load_state_dict(torch.load(final_ckpt, map_location=device))
    test_metrics = evaluate(model, test_loader, device)
    json.dump(test_metrics, open(Path(save_dir) / "test_metrics.json", "w"), indent=2)
    print(f"[Fold {fold_id}] TEST:", test_metrics)
    return test_metrics

# ------------------------
# Main k-fold launcher
# ------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))

    # Ensure dict (some YAML loaders may return string if config path incorrect)
    if isinstance(cfg, str):
        raise SystemExit("Config file parsed to a string — check the YAML file path/contents.")

    # device selection: preserve user choice, else auto-detect
    cfg["device"] = cfg.get("device") or ("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    set_seed(cfg.get("seed", 42))

    print("Using device:", cfg["device"])
    print("Config summary: epoch, batch_size, lr, optimizer:", cfg.get("epochs"), cfg.get("batch_size"), cfg.get("lr"), cfg.get("optimizer", "adamw"))

    splits = Path(cfg["splits_dir"])
    test_csv = splits / "test.csv"

    results = {}
    for fold in range(4):
        fold_dir = splits / f"fold{fold}"
        train_csv = fold_dir / "train.csv"
        val_csv   = fold_dir / "val.csv"
        save_dir  = Path(cfg["save_dir"]) / f"fold{fold}"
        metrics = train_one_fold(cfg, fold, train_csv, val_csv, test_csv, save_dir)
        results[f"fold{fold}"] = metrics

    json.dump(results, open(Path(cfg["save_dir"]) / "results_all_folds.json", "w"), indent=2)
    print("\nAll folds done.\n")

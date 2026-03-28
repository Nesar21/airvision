# kfold_train.py
# 4-fold trainer for LiteFusion (Image+Depth+Numeric)
# Run:
#   source ~/venvs/image_aqi/bin/activate
#   python kfold_train.py --config cfg_kfold.yaml

import os
import json
import time
import argparse
from pathlib import Path

import yaml
import pandas as pd
import torch
from torch.utils.data import DataLoader

from litefusion_model_and_train import (
    AQIMultiDataset,
    collate_batch,
    LiteFusion,
    masked_multi_task_loss,
    evaluate
)


# --------------------------------------
# Train a single fold
# --------------------------------------
def train_one_fold(cfg, fold_id, train_csv, val_csv, test_csv, save_dir):
    device = torch.device(cfg["device"])

    os.makedirs(save_dir, exist_ok=True)
    json.dump(cfg, open(Path(save_dir) / "cfg.json", "w"), indent=2)

    # Loaders
    train_loader = DataLoader(
        AQIMultiDataset(train_csv, img_size=cfg["img_size"]),
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        collate_fn=collate_batch
    )

    val_loader = DataLoader(
        AQIMultiDataset(val_csv, img_size=cfg["img_size"]),
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        collate_fn=collate_batch
    )

    # Model
    model = LiteFusion(
        backbone_name=cfg["backbone"],
        pretrained=cfg["pretrained"],
        img_size=cfg["img_size"]
    ).to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=cfg["epochs"])

    best_val = 1e9

    # --------------------------------------
    # Epoch loop
    # --------------------------------------
    for epoch in range(cfg["epochs"]):
        model.train()
        total_loss = 0
        batches = 0
        t0 = time.time()

        for batch in train_loader:
            img = batch["image"].to(device)
            depth = batch["depth"].to(device)

            pm25_v = torch.nan_to_num(batch["pm25"], nan=0.0).to(device)
            pm10_v = torch.nan_to_num(batch["pm10"], nan=0.0).to(device)
            aqi_bin = torch.zeros(img.shape[0], device=device)

            numeric = torch.stack([pm25_v, pm10_v, aqi_bin], dim=1)

            out = model(img, depth, numeric)

            targets = {
                "aqi": batch["aqi"].to(device),
                "pm25": batch["pm25"].to(device),
                "pm10": batch["pm10"].to(device),
            }

            masks = {
                "has_aqi": batch["has_aqi"].to(device),
                "has_pm25": batch["has_pm25"].to(device),
                "has_pm10": batch["has_pm10"].to(device),
            }

            loss = masked_multi_task_loss(out, targets, masks, cfg)

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

            total_loss += loss.item()
            batches += 1

        sched.step()
        t1 = time.time()

        val_metrics = evaluate(model, val_loader, device)
        val_mae = val_metrics.get("aqi_mae", 1e9)

        print(
            f"[Fold {fold_id}] Epoch {epoch+1}/{cfg['epochs']} | "
            f"train_loss={total_loss/batches:.4f} | val_mae={val_mae:.4f} | time={t1-t0:.1f}s"
        )

        if val_mae < best_val:
            best_val = val_mae
            torch.save(model.state_dict(), Path(save_dir) / "best.pth")

    # --------------------------------------
    # Test evaluation (GLOBAL TEST SET)
    # --------------------------------------
    test_loader = DataLoader(
        AQIMultiDataset(test_csv, img_size=cfg["img_size"]),
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        collate_fn=collate_batch
    )

    model.load_state_dict(torch.load(Path(save_dir) / "best.pth", map_location=device))
    test_metrics = evaluate(model, test_loader, device)

    json.dump(test_metrics, open(Path(save_dir) / "test_metrics.json", "w"), indent=2)
    print(f"[Fold {fold_id}] Test:", test_metrics)

    return test_metrics


# --------------------------------------
# Main K-Fold Launcher
# --------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))

    # Auto select device
    cfg["device"] = (
        "mps" if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    print("Using device:", cfg["device"])

    splits = Path(cfg["splits_dir"])
    results = {}

    # GLOBAL TEST SET (correct fix)
    global_test_csv = splits / "test.csv"

    # Run all 4 folds
    for fold in range(4):
        fold_dir = splits / f"fold{fold}"

        train_csv = fold_dir / "train.csv"
        val_csv   = fold_dir / "val.csv"
        test_csv  = global_test_csv    # FIXED — use single global test set

        save_dir  = Path(cfg["save_dir"]) / f"fold{fold}"

        metrics = train_one_fold(cfg, fold, train_csv, val_csv, test_csv, save_dir)
        results[f"fold{fold}"] = metrics

    # Save all results
    json.dump(results, open(Path(cfg["save_dir"]) / "results_all_folds.json", "w"), indent=2)
    print("All fold results saved.")

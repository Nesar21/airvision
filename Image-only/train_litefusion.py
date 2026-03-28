#!/usr/bin/env python3

import os
import argparse
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from src.data.loaders import build_loaders
from src.model.fusion_model import LiteFusionModel
from src.model.losses import multimodal_loss


def train_one_epoch(model, loader, optimizer, scaler, device):
    model.train()
    total_loss = 0
    count = 0

    for batch in loader:
        img = batch["image"].to(device)
        depth = batch["depth"].to(device)
        aqi = batch["aqi"].to(device)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=True):
            pred = model(img, depth)
            loss = multimodal_loss(pred, aqi)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        count += 1

    return total_loss / count


def validate(model, loader, device):
    model.eval()
    total_mae = 0
    count = 0

    with torch.no_grad():
        for batch in loader:
            img = batch["image"].to(device)
            depth = batch["depth"].to(device)
            aqi = batch["aqi"].to(device)

            pred = model(img, depth)
            mae = torch.abs(pred - aqi).mean()

            total_mae += mae.item()
            count += 1

    return total_mae / count


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_csv = args.train_csv
    val_csv   = args.val_csv
    test_csv  = args.test_csv

    batch_size = args.batch_size

    # 1) Loaders
    train_loader, val_loader, test_loader = build_loaders(
        train_csv, val_csv, test_csv, batch_size=batch_size
    )

    # 2) Model
    model = LiteFusionModel().to(device)

    # 3) Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    best_val = 9999
    os.makedirs("outputs", exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device)
        val_mae = validate(model, val_loader, device)

        print(f"Train Loss: {train_loss:.4f} | Val MAE: {val_mae:.2f}")

        # save checkpoint
        torch.save(model.state_dict(), f"outputs/last.pt")

        if val_mae < best_val:
            best_val = val_mae
            torch.save(model.state_dict(), "outputs/best.pt")
            print("Saved best model.")

    print("\nTraining complete.")
    print(f"Best validation MAE = {best_val:.2f}")

    # --------- Final Test Performance ----------
    test_mae = validate(model, test_loader, device)
    print(f"\nTEST MAE = {test_mae:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--val_csv", required=True)
    parser.add_argument("--test_csv", required=True)

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=15)

    args = parser.parse_args()
    main(args)

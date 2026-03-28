#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import lightgbm as lgb


class TraqidFusionAQIPredictor:
    """
    Night-only AQI predictor for TRAQID Sequence=2.

    Uses:
      - data/traqid_embeddings.csv (numeric + embeddings)
      - models/traqid_night_aqi/best_fusion_model.txt (LightGBM booster)
    """

    def __init__(
        self,
        embeddings_csv: str = "data/traqid_embeddings.csv",
        booster_path: str = "models/traqid_night_aqi/best_fusion_model.txt",
    ):
        if not os.path.exists(embeddings_csv):
            raise FileNotFoundError(f"Embeddings CSV not found: {embeddings_csv}")
        if not os.path.exists(booster_path):
            raise FileNotFoundError(f"LightGBM booster not found: {booster_path}")

        self.embeddings_csv = embeddings_csv
        self.booster_path = booster_path

        # Load data
        df = pd.read_csv(embeddings_csv)

        # Expected numeric + categorical columns
        num_cols = [
            "PM2.5_norm",
            "PM10_norm",
            "Temperature_norm",
            "Humidity_norm",
            "season_idx",
            "day_idx",
        ]
        for c in num_cols:
            if c not in df.columns:
                raise ValueError(f"Required column '{c}' missing in {embeddings_csv}")

        emb_cols = [c for c in df.columns if c.startswith("emb_")]
        if not emb_cols:
            raise ValueError("No embedding columns (emb_*) found in embeddings CSV")

        self.num_cols = num_cols
        self.emb_cols = emb_cols

        # Features and targets
        self.df = df
        self.X = df[num_cols + emb_cols].values.astype("float32")
        self.y = df["aqi"].values.astype("float32")

        # Map from Image id to index for convenience
        if "Image" in df.columns:
            self.image_to_idx = {
                int(row["Image"]): idx
                for idx, row in df.iterrows()
            }
        else:
            self.image_to_idx = {}

        # Load booster
        self.booster = lgb.Booster(model_file=booster_path)

    # -----------------------------------------------------
    # Core prediction helpers
    # -----------------------------------------------------
    def predict_index(self, idx: int):
        """
        Predict AQI for a single row index in the embeddings CSV.
        Returns (pred_aqi, true_aqi).
        """
        if idx < 0 or idx >= len(self.df):
            raise IndexError(f"Index {idx} out of range 0..{len(self.df)-1}")

        X_i = self.X[idx:idx + 1]
        pred = float(self.booster.predict(X_i)[0])
        true = float(self.y[idx])
        return pred, true

    def predict_image_id(self, image_id: int):
        """
        Predict AQI given the TRAQID Image id.
        Requires 'Image' column to exist.
        Returns (pred_aqi, true_aqi).
        """
        if not self.image_to_idx:
            raise RuntimeError("Embeddings CSV has no 'Image' column mapping")

        image_id = int(image_id)
        if image_id not in self.image_to_idx:
            raise KeyError(f"Image id {image_id} not found in embeddings CSV")

        idx = self.image_to_idx[image_id]
        return self.predict_index(idx)

    def predict_all(self):
        """
        Predict AQI for all rows.
        Returns array of predictions with same length as df.
        """
        preds = self.booster.predict(self.X)
        return preds.astype("float32")

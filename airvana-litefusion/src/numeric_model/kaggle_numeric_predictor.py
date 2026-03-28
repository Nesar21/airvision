#!/usr/bin/env python3
"""
kaggle_numeric_predictor.py

Wrapper to use the Kaggle-trained LightGBM AQI model:

- Loads fold boosters from: models/lightgbm_kaggle/aqi_fold{0..4}.txt
- Exposes a simple function to predict AQI from pollutants:
    PM2.5, PM10, NO2, SO2, CO, OZONE, NH3
"""

import os
from typing import Optional, Sequence

import numpy as np
import lightgbm as lgb


KAGGLE_MODEL_DIR = "models/lightgbm_kaggle"


class KaggleNumericAQIPredictor:
    def __init__(self, model_dir: str = KAGGLE_MODEL_DIR) -> None:
        self.boosters = []
        for i in range(5):
            path = os.path.join(model_dir, f"aqi_fold{i}.txt")
            if os.path.exists(path):
                print(f"[KaggleNumeric] Loading {path}")
                self.boosters.append(lgb.Booster(model_file=path))

        if not self.boosters:
            raise FileNotFoundError(
                f"No LightGBM boosters found in {model_dir}. "
                "Train them with train_lightgbm_kaggle_aqi.py first."
            )

        # Order of features must match training
        self.feature_names = ["PM2.5", "PM10", "NO2", "SO2", "CO", "OZONE", "NH3"]

    def _make_feature_vector(
        self,
        pm25: Optional[float] = None,
        pm10: Optional[float] = None,
        no2: Optional[float] = None,
        so2: Optional[float] = None,
        co: Optional[float] = None,
        o3: Optional[float] = None,
        nh3: Optional[float] = None,
    ) -> np.ndarray:
        vals = [pm25, pm10, no2, so2, co, o3, nh3]
        # simple imputation: replace None with 0.0
        vals = [0.0 if v is None else float(v) for v in vals]
        return np.array(vals, dtype=np.float32).reshape(1, -1)

    def predict_aqi(
        self,
        pm25: Optional[float] = None,
        pm10: Optional[float] = None,
        no2: Optional[float] = None,
        so2: Optional[float] = None,
        co: Optional[float] = None,
        o3: Optional[float] = None,
        nh3: Optional[float] = None,
    ) -> float:
        """
        Predict AQI from pollutants using an ensemble of Kaggle boosters.
        Any missing pollutant can be passed as None.
        """
        x = self._make_feature_vector(pm25, pm10, no2, so2, co, o3, nh3)
        preds = [b.predict(x)[0] for b in self.boosters]
        return float(np.mean(preds))

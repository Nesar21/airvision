import torch
from torch.utils.data import Sampler
import numpy as np
import pandas as pd

class NightAwareWeightedSampler(Sampler):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)

        # base weight from master_v2.csv
        base_w = df["sample_weight"].values

        # night boost
        night_boost = df["day_night"].apply(lambda x: 3.0 if x == "Night" else 1.0).values

        # AQI-balanced (inverse freq)
        bin_counts = df["aqi_bin"].value_counts().to_dict()
        bin_w = df["aqi_bin"].map(lambda b: 1.0 / np.sqrt(bin_counts.get(b, 1))).values

        self.weights = torch.tensor(base_w * night_boost * bin_w, dtype=torch.float32)
        self.indices = np.arange(len(df))

    def __iter__(self):
        return iter(
            torch.multinomial(self.weights, len(self.weights), replacement=True).tolist()
        )

    def __len__(self):
        return len(self.weights)

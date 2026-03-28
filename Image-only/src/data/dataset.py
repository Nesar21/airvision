import torch
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import numpy as np
import os

from .transforms import train_rgb_transform, val_rgb_transform, load_depth

class AQIMultiDataset(Dataset):
    """
    Unified dataset for:
        - RGB image
        - depth (optional)
        - aqi, pm25, pm10 (masked labels)
        - day/night
    """

    def __init__(self, csv_path, split="train"):
        self.df = pd.read_csv(csv_path)
        self.split = split
        
        self.is_train = (split == "train")
        self.rgb_tf = train_rgb_transform if self.is_train else val_rgb_transform

        # Pre-extract columns for speed
        self.paths = self.df["image_path"].tolist()
        self.depths = self.df["depth_path"].tolist()
        self.aqi = self.df["aqi"].values
        self.pm25 = self.df["pm25"].values
        self.pm10 = self.df["pm10"].values
        self.has_aqi = self.df["has_aqi"].values.astype(bool)
        self.has_pm25 = self.df["has_pm25"].values.astype(bool)
        self.has_pm10 = self.df["has_pm10"].values.astype(bool)
        self.sample_weights = self.df["sample_weight"].values
        
        self.daynight = self.df["day_night"].fillna("Unknown").tolist()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        p = self.paths[idx]
        d = self.depths[idx]

        # RGB
        img = Image.open(p).convert("RGB")
        img = self.rgb_tf(img)

        # depth
        if isinstance(d, str) and os.path.exists(d):
            depth_tensor = load_depth(d)
        else:
            depth_tensor = torch.zeros(1, img.shape[1], img.shape[2])

        # labels masked
        aqi = torch.tensor(self.aqi[idx], dtype=torch.float32)
        pm25 = torch.tensor(self.pm25[idx], dtype=torch.float32)
        pm10 = torch.tensor(self.pm10[idx], dtype=torch.float32)

        mask_aqi = torch.tensor(self.has_aqi[idx], dtype=torch.bool)
        mask_pm25 = torch.tensor(self.has_pm25[idx], dtype=torch.bool)
        mask_pm10 = torch.tensor(self.has_pm10[idx], dtype=torch.bool)

        return {
            "rgb": img,
            "depth": depth_tensor,
            "aqi": aqi,
            "pm25": pm25,
            "pm10": pm10,
            "mask_aqi": mask_aqi,
            "mask_pm25": mask_pm25,
            "mask_pm10": mask_pm10,
            "sample_weight": torch.tensor(self.sample_weights[idx], dtype=torch.float32),
            "day_night": self.daynight[idx]
        }

from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

from src.utils.env_utils import init_env
from src.utils.manifest import log_run

from .pollutant_filter import compute_pollutant_dominance, compute_subindex_pm
from .temporal_alignment import compute_temporal_conf
from .image_quality import is_low_information_image

from .phase1_constants import (
    DATA_ROOT,
    CONF_MIN_KEEP,
    SOFT_AQI_CONF,
)


def preprocess_phase1():
    # CSV path
    meta_path = (
        DATA_ROOT
        / "Air Pollution Image Dataset"
        / "Air Pollution Image Dataset"
        / "Combined_Dataset"
        / "IND_and_Nep_AQI_Dataset.csv"
    )
    df = pd.read_csv(meta_path)

    # 1) Construct full image path from "Filename"
    img_root = (
        DATA_ROOT
        / "Air Pollution Image Dataset"
        / "Air Pollution Image Dataset"
        / "Combined_Dataset"
        / "All_img"
    )

    df["image_path"] = df["Filename"].apply(
        lambda x: str((img_root / x).resolve())
    )

    # 2) Reconstruct datetime from Year / Month / Day / Hour
    def build_datetime(row):
        hour_str = str(row["Hour"]).strip()  # e.g., '12:00'
        return datetime(
            int(row["Year"]),
            int(row["Month"]),
            int(row["Day"]),
            int(hour_str.split(":")[0]),
            int(hour_str.split(":")[1]),
        )

    df["datetime_image"] = df.apply(build_datetime, axis=1)
    df["datetime_aqi"] = df["datetime_image"]
    df["uses_hourly_measure"] = True

    # 3) No solar elevation → set confidence = 1.0
    df["conf_twilight"] = 1.0

    # 4) PM-dominant filtering
    df = compute_pollutant_dominance(
        df.rename(columns={"PM2.5": "pm25", "PM10": "pm10"})
    )

    pm_dom = df[df["is_pm_dominant"]].copy()
    gas_dom = df[~df["is_pm_dominant"]].copy()

    gas_dom.to_csv("gas_driven_set.csv", index=False)
    df = pm_dom.reset_index(drop=True)

    # 5) Soft AQI label using CPCB mapping
    df["soft_aqi_flag"] = 1
    df["soft_aqi"] = df["pm25"].apply(compute_subindex_pm)
    df["conf_softlabel"] = SOFT_AQI_CONF

    # 6) Temporal confidence (all exact timestamps → strict = 1.0)
    df = compute_temporal_conf(df)

    # 7) Combined confidence
    df["label_confidence"] = df[
        ["conf_temporal", "conf_twilight", "conf_softlabel"]
    ].min(axis=1)

    df = df[df["label_confidence"] >= CONF_MIN_KEEP].reset_index(drop=True)

    # 8) Low-information filter
    df["is_low_information"] = df["image_path"].apply(is_low_information_image)
    df = df[~df["is_low_information"]].reset_index(drop=True)

    # 9) Save canonical metadata
    df.to_csv("metadata_image_only.csv", index=False)
    df.to_csv("metadata_fusion.csv", index=False)

    return df


if __name__ == "__main__":
    device = init_env()
    df = preprocess_phase1()

    log_run(
        phase="phase1_preprocessing",
        stage="full_pipeline_updated",
        description="Phase 1 preprocessing updated for IND+NEP dataset schema.",
        config={"device": str(device)},
        metrics={"rows_final": len(df)},
    )

    print("Phase 1 complete. Rows:", len(df))

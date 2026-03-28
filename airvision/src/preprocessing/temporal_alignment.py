from __future__ import annotations
import pandas as pd
import numpy as np

from .phase1_constants import (
    CONF_TEMP_STRICT,
    CONF_TEMP_LOOSE,
    CONF_TEMP_DAILY,
)

def compute_temporal_conf(df: pd.DataFrame) -> pd.DataFrame:
    """
    df must contain: datetime_image, datetime_aqi, uses_hourly_measure
    """
    deltas = (df["datetime_image"] - df["datetime_aqi"]).dt.total_seconds().abs()
    hours = deltas / 3600.0

    confs = []
    for h in hours:
        if np.isnan(h):
            confs.append(CONF_TEMP_DAILY)
        elif h <= 1:
            confs.append(CONF_TEMP_STRICT)
        elif h <= 6:
            confs.append(CONF_TEMP_LOOSE)
        else:
            confs.append(0.0)
    df["conf_temporal"] = confs
    return df

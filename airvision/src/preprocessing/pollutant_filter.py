from __future__ import annotations

import pandas as pd

def compute_subindex_pm(pm_value: float) -> float:
    """
    Piecewise-linear Indian CPCB ranges for PM2.5.
    """
    if pm_value <= 30: return pm_value * (50 / 30)
    if pm_value <= 60: return 50 + (pm_value - 30) * (50 / 30)
    if pm_value <= 90: return 100 + (pm_value - 60) * (100 / 30)
    if pm_value <= 120: return 200 + (pm_value - 90) * (100 / 30)
    if pm_value <= 250: return 300 + (pm_value - 120) * (100 / 130)
    return 400 + (pm_value - 250) * (100 / 130)


def compute_pollutant_dominance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute PM2.5 & PM10 subindices.
    Mark dominant pollutant row.
    """
    df["sub_pm25"] = df["pm25"].apply(compute_subindex_pm)
    df["sub_pm10"] = df["pm10"].apply(lambda x: compute_subindex_pm(x * 0.7))

    df["dominant"] = df[["sub_pm25", "sub_pm10"]].idxmax(axis=1)
    df["is_pm_dominant"] = df["dominant"].isin(["sub_pm25", "sub_pm10"])
    return df

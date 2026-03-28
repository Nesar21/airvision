from __future__ import annotations

import pathlib

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "kaggle"
PM25VISION_ROOT = PROJECT_ROOT / "pm25vision"

CONF_TEMP_STRICT = 1.0
CONF_TEMP_LOOSE = 0.7
CONF_TEMP_DAILY = 0.5
CONF_MIN_KEEP = 0.3

CONF_TWILIGHT = 0.7
CONF_DAY = 1.0

SOFT_AQI_CONF = 0.5

# Twilight solar elevation bounds
SOLAR_DAY_MIN = 5.0       # degrees
SOLAR_TWILIGHT_MIN = -12  # twilight lower bound
SOLAR_TWILIGHT_MAX = 5    # twilight upper bound

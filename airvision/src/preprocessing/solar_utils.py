from __future__ import annotations
import datetime as dt
from timezonefinder import TimezoneFinder
from pysolar.solar import get_altitude

from .phase1_constants import SOLAR_TWILIGHT_MIN, SOLAR_TWILIGHT_MAX

_tf = TimezoneFinder()


def compute_solar_elevation(lat: float, lon: float, dt_obj: dt.datetime) -> float:
    if dt_obj.tzinfo is None:
        tz = _tf.timezone_at(lat=lat, lng=lon)
        if tz is None:
            return -90.0
        dt_obj = dt_obj.astimezone(dt.timezone.utc)
    return float(get_altitude(lat, lon, dt_obj))


def classify_solar(elev: float) -> str:
    if elev < SOLAR_TWILIGHT_MIN:
        return "night"
    if SOLAR_TWILIGHT_MIN <= elev <= SOLAR_TWILIGHT_MAX:
        return "twilight"
    return "day"

#!/usr/bin/env python3
"""
waqi_log_pollutants.py

Log real-time WAQI data (AQI + pollutants) for multiple cities worldwide.

Run this script periodically (e.g., cron, manual) to accumulate a global
numeric training dataset:

    data/waqi_log.csv

Columns will include:
    timestamp_utc, city, country, lat, lon,
    aqi,
    pm25, pm10, no2, so2, co, o3, nh3
"""

import os
import csv
import argparse
from datetime import datetime, timezone
import requests


# -------------------------------------------------------------------
# City configuration: India + global major cities
# -------------------------------------------------------------------

CITY_CONFIG = [
    # India
    {"city": "Delhi",          "country": "India", "lat": 28.6129, "lon": 77.2295},
    {"city": "Mumbai",         "country": "India", "lat": 19.0760, "lon": 72.8777},
    {"city": "Bengaluru",      "country": "India", "lat": 12.9716, "lon": 77.5946},
    {"city": "Hyderabad",      "country": "India", "lat": 17.3850, "lon": 78.4867},
    {"city": "Chennai",        "country": "India", "lat": 13.0827, "lon": 80.2707},
    {"city": "Kolkata",        "country": "India", "lat": 22.5726, "lon": 88.3639},

    # Global
    {"city": "Beijing",        "country": "China", "lat": 39.9042, "lon": 116.4074},
    {"city": "Shanghai",       "country": "China", "lat": 31.2304, "lon": 121.4737},
    {"city": "Los_Angeles",    "country": "USA",   "lat": 34.0522, "lon": -118.2437},
    {"city": "New_York",       "country": "USA",   "lat": 40.7128, "lon": -74.0060},
    {"city": "London",         "country": "UK",    "lat": 51.5074, "lon": -0.1278},
    {"city": "Paris",          "country": "France","lat": 48.8566, "lon": 2.3522},
    {"city": "Tokyo",          "country": "Japan", "lat": 35.6895, "lon": 139.6917},
    {"city": "Seoul",          "country": "South_Korea","lat": 37.5665, "lon": 126.9780},
    {"city": "Sydney",         "country": "Australia","lat": -33.8688, "lon": 151.2093},
]


def fetch_waqi(lat: float, lon: float, token: str):
    url = f"https://api.waqi.info/feed/geo:{lat};{lon}/?token={token}"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    if data.get("status") != "ok":
        raise RuntimeError(f"WAQI error: {data.get('data')}")

    d = data["data"]
    aqi = d.get("aqi", None)
    iaqi = d.get("iaqi", {}) or {}

    def get_iaqi(key):
        val = iaqi.get(key)
        if isinstance(val, dict):
            return val.get("v")
        return None

    pollutants = {
        "pm25": get_iaqi("pm25"),
        "pm10": get_iaqi("pm10"),
        "no2":  get_iaqi("no2"),
        "so2":  get_iaqi("so2"),
        "co":   get_iaqi("co"),
        "o3":   get_iaqi("o3"),
        "nh3":  get_iaqi("nh3"),
    }

    return aqi, pollutants


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/waqi_log.csv",
                    help="Path to log CSV (default: data/waqi_log.csv)")
    ap.add_argument("--waqi_token", default=None,
                    help="WAQI token (or set WAQI_TOKEN env var)")
    args = ap.parse_args()

    token = args.waqi_token or os.getenv("WAQI_TOKEN")
    if not token:
        raise RuntimeError("Missing WAQI token. Use --waqi_token or set WAQI_TOKEN env.")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    file_exists = os.path.exists(args.out)
    with open(args.out, "a", newline="") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow([
                "timestamp_utc",
                "city",
                "country",
                "lat",
                "lon",
                "aqi",
                "pm25",
                "pm10",
                "no2",
                "so2",
                "co",
                "o3",
                "nh3",
            ])

        now_utc = datetime.now(timezone.utc).isoformat()

        for cfg in CITY_CONFIG:
            city = cfg["city"]
            country = cfg["country"]
            lat = cfg["lat"]
            lon = cfg["lon"]

            try:
                aqi, pollutants = fetch_waqi(lat, lon, token)
            except Exception as e:
                print(f"[WARN] Failed for {city}: {e}")
                continue

            row = [
                now_utc,
                city,
                country,
                lat,
                lon,
                aqi,
                pollutants.get("pm25"),
                pollutants.get("pm10"),
                pollutants.get("no2"),
                pollutants.get("so2"),
                pollutants.get("co"),
                pollutants.get("o3"),
                pollutants.get("nh3"),
            ]
            writer.writerow(row)
            print(f"[OK] Logged {city}: AQI={aqi}, pollutants={pollutants}")

    print(f"\nLog updated → {args.out}")


if __name__ == "__main__":
    main()

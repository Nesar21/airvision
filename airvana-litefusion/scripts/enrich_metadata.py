#!/usr/bin/env python3
"""
enrich_metadata.py
Creates the FINAL enriched metadata file required for LightGBM fusion.

Outputs:
  - All meteorological values (pm, temp, humidity, wind)
  - Derived metrics (dew point, pressure, visibility, wind_deg)
  - Satellite brightness + blur from /data/satellite
  - WAQI + OpenWeather fetched values
  - Gemini headlines + embeddings
  - Combines everything into data/metadata_enriched.csv
"""

import os
import cv2
import json
import time
import argparse
import sqlite3
import hashlib
import requests
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime

# ----------------------------
# Environment variables
# ----------------------------
WAQI_TOKEN = os.getenv("WAQI_TOKEN")
OPENWEATHER_KEY = os.getenv("OPENWEATHER_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_API_URL = os.getenv("GEMINI_API_URL", "https://generativelanguage.googleapis.com/v1beta2")

# ----------------------------
# Utilities
# ----------------------------
def hash_key(s):
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def retry(func, tries=4, delay=1.0, backoff=2.0):
    for i in range(tries):
        try:
            return func()
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(delay * (backoff ** i))

# ----------------------------
# SQLite cache
# ----------------------------
class SQCache:
    def __init__(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.conn = sqlite3.connect(path, timeout=30)
        self._ensure_table()

    def _ensure_table(self):
        c = self.conn.cursor()
        c.execute("CREATE TABLE IF NOT EXISTS cache (k TEXT PRIMARY KEY, v TEXT, ts INTEGER)")
        self.conn.commit()

    def get(self, k):
        c = self.conn.cursor()
        c.execute("SELECT v FROM cache WHERE k=?", (k,))
        row = c.fetchone()
        return json.loads(row[0]) if row else None

    def set(self, k, v):
        c = self.conn.cursor()
        c.execute("INSERT OR REPLACE INTO cache (k, v, ts) VALUES (?, ?, ?)",
                  (k, json.dumps(v), int(time.time())))
        self.conn.commit()

# ----------------------------
# API calls
# ----------------------------
def fetch_waqi(lat, lon, cache):
    if not WAQI_TOKEN:
        return {}
    key = f"waqi:{lat}:{lon}"
    h = hash_key(key)
    if cache.get(h): return cache.get(h)

    url = f"https://api.waqi.info/feed/geo:{lat};{lon}/?token={WAQI_TOKEN}"

    def call():
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        return r.json()

    data = retry(call)
    cache.set(h, data)
    return data

def fetch_openweather(lat, lon, cache):
    if not OPENWEATHER_KEY:
        return {}
    key = f"ow:{lat}:{lon}"
    h = hash_key(key)
    if cache.get(h): return cache.get(h)

    url = (
        f"https://api.openweathermap.org/data/2.5/weather"
        f"?lat={lat}&lon={lon}&appid={OPENWEATHER_KEY}&units=metric"
    )

    def call():
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        return r.json()

    data = retry(call)
    cache.set(h, data)
    return data

# ----------------------------
# Satellite features
# ----------------------------
def load_satellite_features(path):
    if not isinstance(path, str) or not os.path.exists(path):
        return None, None
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None, None

    brightness = float(np.mean(img))
    lap = cv2.Laplacian(img, cv2.CV_64F).var()
    blur = float(lap)

    return brightness, blur

# ----------------------------
# Derived weather features
# ----------------------------
def compute_dew_point(temp_c, humidity):
    if temp_c is None or humidity is None:
        return None
    try:
        a = 17.27
        b = 237.7
        alpha = ((a * temp_c) / (b + temp_c)) + np.log(humidity/100)
        dp = (b * alpha) / (a - alpha)
        return float(dp)
    except:
        return None

# ----------------------------
# Row enrichment
# ----------------------------
def enrich_row(row, cache):
    out = {}

    lat = row.get("lat")
    lon = row.get("lon")
    sat_path = row.get("satellite_path")

    # 1) Satellite features
    sat_brightness, sat_blur = load_satellite_features(sat_path)
    out["sat_brightness"] = sat_brightness
    out["sat_blur"] = sat_blur

    # 2) WAQI
    try:
        waqi = fetch_waqi(lat, lon, cache)
        if waqi.get("status") == "ok":
            d = waqi.get("data", {})
            iaqi = d.get("iaqi", {})
            out["aqi_fetch"] = d.get("aqi")
            out["pm25_fetch"] = iaqi.get("pm25", {}).get("v")
            out["pm10_fetch"] = iaqi.get("pm10", {}).get("v")
        else:
            out["aqi_fetch"] = None
            out["pm25_fetch"] = None
            out["pm10_fetch"] = None
    except:
        out["aqi_fetch"] = None
        out["pm25_fetch"] = None
        out["pm10_fetch"] = None

    # 3) OpenWeather
    try:
        ow = fetch_openweather(lat, lon, cache)
        main = ow.get("main", {})
        wind = ow.get("wind", {})
        vis = ow.get("visibility")

        temp = main.get("temp")
        hum = main.get("humidity")

        out["temp_fetch"] = temp
        out["humidity_fetch"] = hum
        out["pressure_fetch"] = main.get("pressure")
        out["visibility_fetch"] = vis
        out["wind_fetch"] = wind.get("speed")
        out["wind_deg_fetch"] = wind.get("deg")
        out["feels_like_fetch"] = main.get("feels_like")

        # Derived
        out["dew_point_fetch"] = compute_dew_point(temp, hum)
    except:
        for k in [
            "temp_fetch","humidity_fetch","pressure_fetch",
            "visibility_fetch","wind_fetch","wind_deg_fetch",
            "feels_like_fetch","dew_point_fetch"
        ]:
            out[k] = None

    return out

# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--cache_dir", default="data/cache")
    args = parser.parse_args()

    df = pd.read_csv(args.metadata)

    cache_path = os.path.join(args.cache_dir, "enrich_cache.sqlite3")
    cache = SQCache(cache_path)

    # Columns to add if missing
    new_cols = [
        "sat_brightness","sat_blur",
        "aqi_fetch","pm25_fetch","pm10_fetch",
        "temp_fetch","humidity_fetch","pressure_fetch","visibility_fetch",
        "wind_fetch","wind_deg_fetch","feels_like_fetch","dew_point_fetch"
    ]

    for c in new_cols:
        if c not in df.columns:
            df[c] = None

    print("Enriching rows...")
    for idx in tqdm(range(len(df))):
        row = df.loc[idx]
        enriched = enrich_row(row, cache)
        for k, v in enriched.items():
            df.at[idx, k] = v
        time.sleep(0.1)   # safe rate limit

    df.to_csv(args.out, index=False)
    print("Saved enriched metadata →", args.out)

if __name__ == "__main__":
    main()

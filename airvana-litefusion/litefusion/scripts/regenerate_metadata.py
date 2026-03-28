import pandas as pd
import sqlite3
import argparse

def load_cache():
    conn = sqlite3.connect("data/cache/enrich_cache.sqlite3")
    return conn

def restore_wx_features(df, conn):
    wx = pd.read_sql_query("SELECT * FROM weather_cache", conn)
    return df.merge(wx, on="image_path", how="left")

def restore_sat_features(df, conn):
    sat = pd.read_sql_query("SELECT * FROM satellite_cache", conn)
    return df.merge(sat, on="image_path", how="left")

def restore_aqi(df, conn):
    aqi = pd.read_sql_query("SELECT * FROM aqi_cache", conn)
    return df.merge(aqi, on="image_path", how="left")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile", required=True)
    parser.add_argument("--outfile", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.infile)
    conn = load_cache()

    print("Restoring WAQI continuous AQI...")
    df = restore_aqi(df, conn)

    print("Restoring weather features...")
    df = restore_wx_features(df, conn)

    print("Restoring satellite haze features...")
    df = restore_sat_features(df, conn)

    print("Saving enriched file →", args.outfile)
    df.to_csv(args.outfile, index=False)
    print("DONE.")

if __name__ == "__main__":
    main()

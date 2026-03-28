import pandas as pd
from pathlib import Path

splits_root = Path("splits")

def pick(df, base):
    if base in df.columns:
        return base
    cand = [c for c in df.columns if c.startswith(base + "_")]
    if len(cand) == 0:
        raise KeyError(f"Missing column: {base}")
    return cand[0]

for k in range(5):
    for split in ["train", "val"]:
        path = splits_root / f"fold{k}_{split}.csv"
        df = pd.read_csv(path)

        col_aqi = pick(df, "AQI")
        col_class = pick(df, "AQI_Class")
        col_img = pick(df, "image_path")
        col_conf = pick(df, "label_confidence")

        clean = pd.DataFrame({
            "image_id": df["image_id"],
            "image_path": df[col_img],
            "AQI": pd.to_numeric(df[col_aqi], errors="coerce"),
            "AQI_Class": df[col_class].astype(str),
            "label_confidence": pd.to_numeric(df[col_conf], errors="coerce"),
            "city": df["city"].astype(str),
        })

        clean["aqi_bin"] = pd.cut(
            clean["AQI"],
            bins=[0,50,100,150,200,300,10000],
            labels=False,
            include_lowest=True
        )

        clean.to_csv(path, index=False)
        print("CLEANED:", path)

import os
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.fusion.litefusion_api import LiteFusionPredictor
from src.image_model.mobilenet_predictor import MobileNetAQIPredictor
import lightgbm as lgb

def rmse(y, yhat):
    return np.sqrt(mean_squared_error(y, yhat))


def load_lightgbm_models():
    boosters = []
    for i in range(5):
        path = f"models/lightgbm/weather_fold{i}.txt"
        if os.path.exists(path):
            boosters.append(lgb.Booster(model_file=path))
    return boosters


def predict_lightgbm(boosters, X):
    if not boosters:
        return np.zeros(len(X))
    preds = []
    for booster in boosters:
        preds.append(booster.predict(X))
    return np.mean(preds, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata_test", required=True)
    parser.add_argument("--results_dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    print("Loading metadata...")
    df = pd.read_csv(args.metadata_test)

    df["image_path"] = df["image_path"].apply(
        lambda x: x if "data/images/128_test/" in x else os.path.join("data/images/128_test", str(x))
    )
    y_true = df["aqi"].astype(float).values

    print("Loading image ensemble...")
    model_paths = [
        "models/mobilenet/mnv3_fold0.pt",
        "models/mobilenet/mnv3_fold1.pt",
        "models/mobilenet/mnv3_fold2.pt",
        "models/mobilenet/mnv3_fold3.pt",
        "models/mobilenet/mnv3_fold4.pt",
    ]
    img_models = [MobileNetAQIPredictor(p, device="mps") for p in model_paths]

    def predict_img(paths):
        preds = []
        for m in img_models:
            fold_preds = []
            for p in paths:
                pr, _ = m.predict(p)
                fold_preds.append(pr)
            preds.append(fold_preds)
        return np.mean(np.array(preds), axis=0)

    y_img = predict_img(df["image_path"].tolist())

    feature_cols = [
        "pm25_fetch","pm10_fetch","temp_fetch","humidity_fetch",
        "pressure_fetch","visibility_fetch","wind_fetch","wind_deg_fetch",
        "feels_like_fetch","dew_point_fetch",
        "sat_brightness","sat_blur","sat_color_skew",
    ]
    df_X = df[feature_cols].fillna(df[feature_cols].median())

    print("Numeric prediction...")
    boosters = load_lightgbm_models()
    y_lgb = predict_lightgbm(boosters, df_X.values)

    print("News placeholder...")
    df["news"] = df["news"].fillna("")
    y_news = np.zeros(len(df)) + 50.0  # no text model yet

    print("Naive RMSE fusion...")
    w_img = 1 / (rmse(y_true, y_img)**2)
    w_lgb = 1 / (rmse(y_true, y_lgb)**2)
    w_news = 1 / (100.0**2)
    w_sum = w_img + w_lgb + w_news
    y_naive = (w_img*y_img + w_lgb*y_lgb + w_news*y_news) / w_sum

    print("LiteFusion...")
    fusion = LiteFusionPredictor()
    y_lite = []
    for i in range(len(df)):
        fused = fusion.fuse_aqi(
            image=(y_img[i], 8.0),
            numeric=(y_lgb[i], 12.0),
            news=(y_news[i], 0.5),
        )
        y_lite.append(fused.mean)
    y_lite = np.array(y_lite)

    def write(name, y):
        mae = mean_absolute_error(y_true, y); rm = rmse(y_true, y); r2 = r2_score(y_true, y)
        print(f"{name}: MAE={mae:.3f}, RMSE={rm:.3f}, R2={r2:.3f}")
        with open(os.path.join(args.results_dir, f"testset_{name}_metrics.txt"),"w") as f:
            f.write(f"MAE : {mae:.3f}\nRMSE: {rm:.3f}\nR2  : {r2:.3f}\n")

    write("image_only", y_img)
    write("naive_fusion", y_naive)
    write("litefusion", y_lite)

    pd.DataFrame({
        "true": y_true,
        "img": y_img,
        "numeric": y_lgb,
        "news": y_news,
        "naive": y_naive,
        "lite": y_lite
    }).to_csv(os.path.join(args.results_dir,"testset_fusion_predictions.csv"), index=False)

    print("\n✓ Test fusion complete.")
    print("Results saved to:", args.results_dir)


if __name__ == "__main__":
    main()

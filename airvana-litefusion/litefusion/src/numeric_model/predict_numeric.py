import numpy as np
import lightgbm as lgb

FEATURES = [
    "pm25_fetch","pm10_fetch","temp_fetch","humidity_fetch","pressure_fetch",
    "visibility_fetch","wind_fetch","wind_deg_fetch",
    "feels_like_fetch","dew_point_fetch",
    "sat_brightness","sat_blur","sat_color_skew"
]

def load_numeric_models():
    boosters = []
    for i in range(5):
        path = f"models/lightgbm/weather_fold{i}.txt"
        try:
            boosters.append(lgb.Booster(model_file=path))
        except:
            pass
    return boosters

def predict_numeric(df, boosters):
    X = df[FEATURES].astype(float)
    X = X.fillna(X.median(numeric_only=True))
    preds = []
    for b in boosters:
        preds.append(b.predict(X))
    return np.mean(np.vstack(preds), axis=0)

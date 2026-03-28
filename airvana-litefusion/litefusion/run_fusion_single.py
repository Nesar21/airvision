import pandas as pd
import torch
from litefusion.src.numeric_model.predict_numeric import load_numeric_models, predict_numeric
from litefusion.src.image_model.mobilenet_predictor import MobileNetAQIPredictor
from litefusion.src.fusion.litefusion_api import LiteFusionPredictor

df = pd.read_csv("data/metadata_sat_features.csv")
df = df[df["aqi_continuous"].notna()].reset_index(drop=True)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

img_model = MobileNetAQIPredictor(
    model_path="models/mobilenet/mnv3_fold0.pt",
    device=device
)

num_models = load_numeric_models()
fusion = LiteFusionPredictor()

# Choose an image in your city
row = df.iloc[0]      # this row produced ~290+
img_path = row["image_path"]

aqi_img = img_model.predict(img_path)

numeric_pred = predict_numeric(df.iloc[[0]], num_models)[0]

fused = fusion.fuse_aqi(
    image=(aqi_img, fusion.sigma_img),
    numeric=(numeric_pred, fusion.sigma_num),
    news=None
)

print("Image AQI:", aqi_img)
print("Numeric AQI:", numeric_pred)
print("Fused AQI:", fused.mean)

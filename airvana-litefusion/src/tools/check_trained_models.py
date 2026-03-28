import os
import torch
from torchvision import transforms
from PIL import Image
import traceback

# Safe imports – only classes that actually exist
try:
    from src.image_model.mobilenet_predictor import MobileNetAQIPredictor
except:
    MobileNetAQIPredictor = None

try:
    from src.image_model.mobilenetv3_traqid_predictor import MobileNetTraqidPredictor
except:
    MobileNetTraqidPredictor = None

try:
    from src.image_model.haze_predictor import HazeModelPredictor
except:
    HazeModelPredictor = None


MODEL_DIRS = [
    "models/mobilenet",
    "models/traqid_night_aqi",
    "models/mobilenet_haze",
    "models/mobilenet_haze_multisource",
    "models/pm25"
]

DUMMY_IMG = "data/images/manual_test/Delhi-Air-Pollution-1-1.jpg"


def load_img():
    img = Image.open(DUMMY_IMG).convert("RGB")
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    return tf(img).unsqueeze(0)


def test_model(model_path):
    print("\n" + "=" * 100)
    print(f"TESTING MODEL → {model_path}")
    print("=" * 100)

    candidates = []

    if MobileNetAQIPredictor:
        candidates.append(("MobileNetAQIPredictor", MobileNetAQIPredictor))

    if MobileNetTraqidPredictor:
        candidates.append(("MobileNetTraqidPredictor", MobileNetTraqidPredictor))

    if HazeModelPredictor:
        candidates.append(("HazeModelPredictor", HazeModelPredictor))

    dummy = load_img()

    matched = False

    for name, ctor in candidates:
        print(f"\nTrying → {name}")

        try:
            model = ctor(model_path)
            out = model.predict_tensor(dummy)
            print(f"SUCCESS [{name}] → Output: {out}")
            matched = True
            break
        except Exception as e:
            print(f"FAILED [{name}] → {e}")
            print(traceback.format_exc())

    if not matched:
        print("\nNO MATCHING ARCHITECTURE FOR THIS WEIGHT FILE.")


def main():
    all_models = []

    for d in MODEL_DIRS:
        full = os.path.join(os.getcwd(), d)
        if not os.path.exists(full):
            continue
        for f in os.listdir(full):
            if f.endswith(".pt"):
                all_models.append(os.path.join(full, f))

    print(f"FOUND {len(all_models)} MODELS.\n")

    for model_path in all_models:
        test_model(model_path)


if __name__ == "__main__":
    main()

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os


class MobileNetTraqidPredictor(nn.Module):
    """
    Predictor for mnv3_regression.pt
    (trained using train_mobilenetv3_traqid_night_regression.py)
    """

    def __init__(self, model_path: str):
        super().__init__()

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Missing model checkpoint: {model_path}")

        # Model architecture must match training file
        from mobilenet.mobilenetv3_haze import mobilenet_v3_large

        self.model = mobilenet_v3_large(num_classes=1)   # regression head
        state = torch.load(model_path, map_location="cpu")
        self.model.load_state_dict(state)
        self.model.eval()

        # Preprocessing (same as training)
        self.tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def preprocess(self, img_path):
        img = Image.open(img_path).convert("RGB")
        return self.tf(img).unsqueeze(0)

    def predict(self, img_path):
        x = self.preprocess(img_path)
        with torch.no_grad():
            pred = self.model(x).item()
        return float(pred)

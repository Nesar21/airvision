import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small
from PIL import Image
import numpy as np

class MobileNetAQIPredictor:
    def __init__(self, model_path, device):
        self.device = device

        base = mobilenet_v3_small(weights=None)
        base.classifier[3] = nn.Linear(base.classifier[3].in_features, 1)
        self.model = base.to(device)

        state = torch.load(model_path, map_location=device)
        self.model.load_state_dict(state)
        self.model.eval()

        self.mean = torch.tensor([0.485,0.456,0.406]).view(3,1,1).to(device)
        self.std = torch.tensor([0.229,0.224,0.225]).view(3,1,1).to(device)

    def preprocess(self, img_path):
        img = Image.open(img_path).convert("RGB").resize((128,128))
        x = torch.tensor(np.array(img)).float()/255.0
        x = x.permute(2,0,1).to(self.device)
        x = (x - self.mean) / self.std
        return x.unsqueeze(0)

    def predict(self, img_path):
        x = self.preprocess(img_path)
        with torch.no_grad():
            y = self.model(x).item()
        return float(y)

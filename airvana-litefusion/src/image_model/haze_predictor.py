import os
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import mobilenet_v3_small


class MobileNetHazePredictor:
    """
    Ensemble haze/smog classifier.

    Label convention:
      0 -> CLEAR
      1 -> HAZY/SMOG
    """
    def __init__(self, model_paths, device="cpu", img_size=128):
        self.device = torch.device(device)
        self.models = []

        # Same kind of preprocessing as your other image models
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        if isinstance(model_paths, str):
            model_paths = [model_paths]

        # Build architecture EXACTLY like training script and load weights
        for path in model_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Haze model not found: {path}")

            # 1) base mobilenet_v3_small
            model = mobilenet_v3_small(weights=None)

            # 2) training script did:
            #    in_features = model.classifier[3].in_features
            #    model.classifier[3] = nn.Linear(in_features, 2)
            in_features = model.classifier[3].in_features
            model.classifier[3] = nn.Linear(in_features, 2)

            # 3) load saved state
            state = torch.load(path, map_location=self.device)
            missing, unexpected = model.load_state_dict(state, strict=False)
            if missing or unexpected:
                print(f"[WARN] When loading {path}:")
                print("  missing keys   :", missing)
                print("  unexpected keys:", unexpected)

            model.to(self.device)
            model.eval()
            self.models.append(model)

        if not self.models:
            raise RuntimeError("No haze models could be loaded")

    def _prep(self, img_path: str) -> torch.Tensor:
        img = Image.open(img_path).convert("RGB")
        x = self.transform(img).unsqueeze(0)  # [1, C, H, W]
        return x.to(self.device)

    def predict(self, img_path):
        """
        Returns:
          label: 0 (CLEAR) or 1 (HAZY/SMOG)
          confidence: probability of predicted label
        """
        x = self._prep(img_path)

        logits_sum = 0
        with torch.no_grad():
            for m in self.models:
                logits_sum = logits_sum + m(x)

        logits = logits_sum / len(self.models)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]

        label = int(probs.argmax())
        confidence = float(probs[label])
        return label, confidence

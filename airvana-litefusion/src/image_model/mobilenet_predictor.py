import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import random

class MobileNetAQIPredictor:
    """
    MobileNet AQI predictor with optional MC-Dropout uncertainty.
    Works with your existing trained weights.
    """

    def __init__(self, model_path, device=None, num_mc_samples=15):
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.num_mc_samples = num_mc_samples

        # Load your trained MobileNetV3 model
        checkpoint = torch.load(model_path, map_location=self.device)

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            # Your training script might have saved state dict inside a wrapper
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        # Build MobileNetV3 with regression head
        from torchvision.models import mobilenet_v3_small
        self.model = mobilenet_v3_small(pretrained=False)
        self.model.classifier[3] = nn.Linear(self.model.classifier[3].in_features, 1)

        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(self.device)
        self.model.eval()

        # Enable dropout even in eval mode
        self._enable_dropout(self.model)

        # Standard transforms
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def _enable_dropout(self, module):
        """
        Recursively enables dropout layers during inference.
        """
        if isinstance(module, nn.Dropout):
            module.train()
        for child in module.children():
            self._enable_dropout(child)

    def _forward_once(self, image_tensor):
        """
        Forward pass returning a single AQI prediction.
        """
        with torch.no_grad():
            out = self.model(image_tensor)
            return out.item()

    def predict(self, image_path, return_uncertainty=False, raw_image=None):
        """
        Returns:
            if return_uncertainty = False:
                float → AQI estimate

            if return_uncertainty = True:
                (mu, sigma, metadata)
        """

        # Load image
        if raw_image is not None:
            img = raw_image
        else:
            img = Image.open(image_path).convert("RGB")

        x = self.transform(img).unsqueeze(0).to(self.device)

        # Normal simple prediction
        if not return_uncertainty:
            mu = self._forward_once(x)
            return mu

        # MC Dropout: multiple forward passes
        preds = []
        for _ in range(self.num_mc_samples):
            # Add tiny noise to input for better uncertainty
            noise = torch.randn_like(x) * 0.001
            preds.append(self._forward_once(x + noise))

        preds = np.array(preds)
        mu = float(np.mean(preds))
        sigma = float(np.std(preds))

        metadata = {
            "mc_samples": self.num_mc_samples,
            "min": float(np.min(preds)),
            "max": float(np.max(preds)),
        }

        return mu, sigma, metadata

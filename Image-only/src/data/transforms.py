import torch
import torchvision.transforms as T
from PIL import Image

# Base transformation shared between RGB and depth
IMG_SIZE = 224

train_rgb_transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
])

val_rgb_transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
])

# depth transform (keeps depth as 1-channel float tensor)
def load_depth(path):
    import numpy as np
    arr = np.load(path)
    arr = arr.astype("float32")
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)
    return torch.from_numpy(arr).unsqueeze(0)

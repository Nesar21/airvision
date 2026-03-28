import os
import torch
from pprint import pprint

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT, "..", "models")

def inspect_state_dict(path):
    print("\n" + "="*80)
    print(f"INSPECTING: {path}")
    print("="*80)

    try:
        sd = torch.load(path, map_location="cpu")
    except Exception as e:
        print(f"FAILED TO LOAD: {e}")
        return

    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]

    if not isinstance(sd, dict):
        print("This file is not a state_dict.")
        return

    layer_shapes = {k: tuple(v.shape) for k, v in sd.items() if hasattr(v, "shape")}

    # Print basic info
    print("\nLAYER COUNT:", len(layer_shapes))

    # Detect architecture clues
    arch = "UNKNOWN"
    clues = []

    keys = list(layer_shapes.keys())

    if any("haze" in k.lower() for k in keys):
        clues.append("Haze model head detected")

    if any("pm25" in k.lower() for k in keys):
        clues.append("PM2.5 regression head detected")

    if any("features.1.block" in k for k in keys):
        clues.append("Looks like custom MobileNetV3 block definitions (NOT torchvision)")

    if any("classifier" in k for k in keys):
        out_dim = layer_shapes[[k for k in keys if "classifier" in k][-1]][0]
        clues.append(f"Classifier output dim approx → {out_dim}")

    print("\nARCHITECTURE CLUES:")
    for c in clues:
        print("  •", c)

    # Print the first 20 layers for visual inspection
    print("\nFIRST 20 LAYERS:")
    for i, (k, v) in enumerate(layer_shapes.items()):
        print(f"{i:03d}: {k} → {v}")
        if i >= 19:
            break

    print("\nSUGGESTED MATCHES:")

    # Heuristic matching to training scripts
    def match(pattern):
        return any(pattern in k for k in keys)

    if match("traqid") or match("night"):
        print("  → Likely trained by: train_mobilenetv3_traqid_night_regression.py")
    elif match("pm25"):
        print("  → Likely trained by: train_mobilenetv3_pm25.py")
    elif match("haze"):
        print("  → Likely trained by: train_mobilenetv3_haze_kfold.py OR haze_multisource")
    elif match("features.1.block.2"):
        print("  → Strong indication: train_mobilenetv3_kfold.py (custom architecture)")
    else:
        print("  → Unknown: Manual inspection needed")

def main():
    print("SCANNING:", MODELS_DIR)
    for root, dirs, files in os.walk(MODELS_DIR):
        for f in files:
            if f.endswith(".pt") or f.endswith(".pth") or f.endswith(".bin"):
                inspect_state_dict(os.path.join(root, f))

if __name__ == "__main__":
    main()

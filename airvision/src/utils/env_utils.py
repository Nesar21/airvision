from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np
import torch

from src.config import GLOBAL_SEED, USE_FLOAT32_ONLY


def _safe_set_pythonhashseed(seed: int) -> None:
    """Ensure PYTHONHASHSEED is safe and valid."""
    try:
        seed_int = int(seed)
        if seed_int < 0 or seed_int > 4294967295:
            raise ValueError
        os.environ["PYTHONHASHSEED"] = str(seed_int)
    except Exception:
        os.environ["PYTHONHASHSEED"] = "0"


def set_global_seed(seed: int = GLOBAL_SEED) -> None:
    """Set RNG seeds for Python, NumPy, PyTorch."""
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    _safe_set_pythonhashseed(seed)

    torch.use_deterministic_algorithms(True, warn_only=True)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def configure_float32() -> None:
    """Force float32-only."""
    if USE_FLOAT32_ONLY:
        try:
            torch.set_default_dtype(torch.float32)
        except Exception:
            pass

        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass


def get_device(prefer_mps: bool = True) -> torch.device:
    """Return device with priority: MPS > CUDA > CPU."""
    if prefer_mps and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def enable_anomaly_detection(enabled: bool) -> None:
    torch.autograd.set_detect_anomaly(enabled)


def init_env(
    seed: int = GLOBAL_SEED,
    prefer_mps: bool = True,
    anomaly_for_debug: bool = False,
) -> torch.device:
    """Main environment initializer for all phases."""
    set_global_seed(seed)
    configure_float32()
    enable_anomaly_detection(anomaly_for_debug)

    device = get_device(prefer_mps=prefer_mps)

    print(f"[ENV] Seed set to {seed}")
    print(f"[ENV] Float32-only: {USE_FLOAT32_ONLY}")
    print(f"[ENV] Device: {device}")
    print(f"[ENV] Anomaly detection: {anomaly_for_debug}")

    return device


if __name__ == "__main__":
    dev = init_env()
    x = torch.randn(2, 3, device=dev)
    y = x @ torch.randn(3, 4, device=dev)
    print("Sanity matmul OK, shape:", y.shape)

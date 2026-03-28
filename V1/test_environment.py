"""Test preprocessing environment"""
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from multiprocessing import Pool, cpu_count

print("=" * 60)
print("Environment Check")
print("=" * 60)

# Check versions
print(f"\nOpenCV version: {cv2.__version__}")
print(f"NumPy version: {np.__version__}")
print(f"Pandas version: {pd.__version__}")

# Check CPU cores
print(f"\nCPU cores available: {cpu_count()}")

# Test basic CV operations
test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
print(f"\nCV2 color conversion: OK")

# Test FFT
f = np.fft.fft2(gray)
print(f"NumPy FFT: OK")

# Test file paths
test_path = Path('/Users/nesar/VS/V1/')
print(f"\nBase directory exists: {test_path.exists()}")

print("\n✅ All checks passed! Ready to preprocess.")
print("=" * 60)

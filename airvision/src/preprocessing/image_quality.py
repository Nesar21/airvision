from __future__ import annotations

import cv2
import numpy as np
import pathlib

def is_low_information_image(path: str | pathlib.Path) -> bool:
    """
    Simple heuristic: very low variance OR extreme brightness OR empty image.
    """
    img = cv2.imread(str(path))
    if img is None:
        return True

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if gray.size == 0:
        return True

    var = gray.var()
    mean = gray.mean()

    if var < 5:      # extremely low detail
        return True
    if mean < 10:    # too dark
        return True
    if mean > 245:   # too bright
        return True

    return False

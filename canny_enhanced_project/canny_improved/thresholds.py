"""Choosing the low and high thresholds for double thresholding.

fixed   use the values given by the user.
median  classic automatic rule: low = (1 - s) * m, high = (1 + s) * m, where m is
        the median grey level of the smoothed image and s = 0.33 by default.
        It is a heuristic and assumes the image is reasonably exposed.
otsu    Otsu's method on the non-zero gradient magnitudes left after non-maximum
        suppression gives the high threshold, and low = 0.5 * high.
"""
from __future__ import annotations

import numpy as np

MODES = ("fixed", "median", "otsu")


def select_thresholds(mode, smoothed, nms, low=20.0, high=40.0, s=0.33):
    if mode == "fixed":
        return float(low), float(high)
    if mode == "median":
        m = float(np.median(smoothed))
        return max(0.0, (1 - s) * m), min(255.0, (1 + s) * m)
    if mode == "otsu":
        vals = nms[nms > 0]
        if vals.size < 2 or vals.min() == vals.max():
            return float(low), float(high)          # nothing to split, fall back
        from skimage.filters import threshold_otsu

        hi = float(threshold_otsu(vals))
        return 0.5 * hi, hi
    raise ValueError(f"unknown threshold mode '{mode}', choose from {MODES}")

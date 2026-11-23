"""Noise reduction methods that can replace the Gaussian step of Canny.

Gaussian and median use NumPy/SciPy. The bilateral filter is written out in
NumPy here (checked against OpenCV in the tests). Non-local means calls
OpenCV's fastNlMeansDenoising, because a plain NumPy version would be far too
slow to be useful.
All methods take and return a float grayscale array of the same size.
"""
from __future__ import annotations

import numpy as np
import scipy.ndimage as ndi

from .stages import smooth_gaussian


def gaussian(gray, sigma=1.0, kernel="standard", **_):
    return smooth_gaussian(gray, sigma, kernel)


def median(gray, size=3, **_):
    """Replaces each pixel by the median of a size x size window. Good against salt and pepper noise."""
    return ndi.median_filter(gray, size=size, mode="reflect")


def bilateral(gray, radius=2, sigma_space=2.0, sigma_color=25.0, **_):
    """Edge-preserving smoothing: neighbours count less when they are far away or very different in brightness."""
    pad = np.pad(gray, radius, mode="reflect")
    h, w = gray.shape
    num = np.zeros_like(gray, dtype=np.float64)
    den = np.zeros_like(gray, dtype=np.float64)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            nb = pad[radius + dy:radius + dy + h, radius + dx:radius + dx + w]
            wgt = np.exp(-(dy * dy + dx * dx) / (2 * sigma_space ** 2)
                         - (nb - gray) ** 2 / (2 * sigma_color ** 2))
            num += wgt * nb
            den += wgt
    return num / den


def nlm(gray, h=10.0, template=7, search=21, **_):
    """Non-local means (OpenCV). Averages pixels whose surrounding patches look alike."""
    import cv2

    u8 = np.clip(np.rint(gray), 0, 255).astype(np.uint8)
    return cv2.fastNlMeansDenoising(u8, None, float(h), int(template), int(search)).astype(np.float64)


def none(gray, **_):
    return gray.copy()


METHODS = {"gaussian": gaussian, "median": median, "bilateral": bilateral, "nlm": nlm, "none": none}


def denoise(gray, method="gaussian", **params):
    if method not in METHODS:
        raise ValueError(f"unknown denoise method '{method}', choose from {sorted(METHODS)}")
    return METHODS[method](gray, **params)

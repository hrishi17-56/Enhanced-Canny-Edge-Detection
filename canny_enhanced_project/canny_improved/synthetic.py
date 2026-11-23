"""Synthetic test images with known edges, and ways to degrade any image.

Real photographs have no ground-truth edge map in this project, so accuracy
(precision, recall, F1) is measured on drawn scenes whose region boundaries are
known exactly. Drawn scenes are easier than photographs, so scores here say
nothing about performance on natural images.
"""
from __future__ import annotations

import numpy as np
from skimage import draw
from skimage.segmentation import find_boundaries


def synthetic_scene(size=256, seed=0):
    """Return (uint8 image, uint8 ground-truth edge map) with rectangles, discs and an ellipse."""
    rng = np.random.default_rng(seed)
    labels = np.zeros((size, size), dtype=np.int32)
    levels = {0: int(rng.integers(60, 100))}
    for i in range(1, 7):
        kind = i % 3
        cy, cx = rng.integers(size // 6, size - size // 6, 2)
        r = int(rng.integers(size // 12, size // 6))
        if kind == 0:
            rr, cc = draw.rectangle((max(cy - r, 0), max(cx - r, 0)), extent=(2 * r, int(1.5 * r)), shape=labels.shape)
        elif kind == 1:
            rr, cc = draw.disk((cy, cx), r, shape=labels.shape)
        else:
            rr, cc = draw.ellipse(cy, cx, r, int(0.6 * r), shape=labels.shape, rotation=float(rng.uniform(0, 3)))
        labels[rr, cc] = i
        levels[i] = int(rng.integers(130, 250)) if i % 2 else int(rng.integers(20, 55))
    img = np.zeros(labels.shape, np.float64)
    for k, v in levels.items():
        img[labels == k] = v
    gt = (find_boundaries(labels, mode="inner") * 255).astype(np.uint8)
    return np.clip(img, 0, 255).astype(np.uint8), gt


def add_noise(img, sigma, seed=0):
    rng = np.random.default_rng(seed)
    return np.clip(img.astype(np.float64) + rng.normal(0, sigma, img.shape), 0, 255)


def change_contrast(img, factor):
    """Scale contrast around the mean grey level. factor < 1 flattens the image."""
    a = img.astype(np.float64)
    return np.clip((a - a.mean()) * factor + a.mean(), 0, 255)


def change_brightness(img, gamma):
    """Gamma change. gamma < 1 brightens, gamma > 1 darkens."""
    return 255.0 * (np.clip(img.astype(np.float64), 0, 255) / 255.0) ** gamma


def resize(img, size):
    import PIL.Image

    return np.asarray(PIL.Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).resize((size, size), PIL.Image.BILINEAR),
                      dtype=np.float64)

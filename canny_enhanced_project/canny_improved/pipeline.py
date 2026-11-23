"""Improved manual Canny: configurable noise reduction and threshold selection."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from . import stages as S
from .denoise import denoise as run_denoise
from .thresholds import select_thresholds


@dataclass
class CannyResult:
    edges: np.ndarray                 # uint8, 0 or 255, same size as the input
    thresholds: tuple                 # (low, high) actually used
    timings: dict                     # milliseconds per step
    stages: dict = field(default_factory=dict)   # intermediate images when keep_stages=True


def canny_improved(image, denoise="gaussian", sigma=1.0, denoise_params=None,
                   threshold="fixed", low=20.0, high=40.0, hysteresis="single_pass",
                   quantize="round", nms_ties="keep_one", kernel="standard", keep_stages=False) -> CannyResult:
    """Run the six stages on an image path, PIL image or array.

    denoise     gaussian, median, bilateral, nlm or none
    sigma       Gaussian standard deviation in pixels (when denoise == "gaussian")
    kernel      "standard" (true Gaussian) or "original" (the original 7x7 kernel)
    threshold   fixed, median or otsu (see thresholds.py)
    """
    t = {}

    def step(name, fn, *a, **k):
        t0 = time.perf_counter()
        out = fn(*a, **k)
        t[name] = (time.perf_counter() - t0) * 1000.0
        return out

    gray = step("grayscale", S.to_gray, image)
    if min(gray.shape) < 8:
        raise ValueError("image must be at least 8 pixels on each side")
    params = dict(denoise_params or {})
    if denoise == "gaussian":
        params.setdefault("sigma", sigma)
        params.setdefault("kernel", kernel)
    smooth = step("smoothing", run_denoise, gray, denoise, **params)
    smooth = S.quantize(smooth, quantize)
    sx, sy = step("sobel", S.sobel, smooth)
    mag, direction = step("magnitude", S.magnitude_direction, sx, sy)
    nms = step("nms", S.non_max_suppression, mag, direction, nms_ties)
    lo, hi = step("thresholds", select_thresholds, threshold, smooth, nms, low, high)
    thr = step("double_threshold", S.double_threshold, nms, lo, hi)
    edges = step("hysteresis", S.hysteresis, thr, hysteresis)
    t["total"] = sum(t.values())
    st = {}
    if keep_stages:
        st = {"gray": gray, "smoothed": smooth, "magnitude": mag, "nms": nms, "thresholded": thr}
    return CannyResult(edges, (lo, hi), t, st)

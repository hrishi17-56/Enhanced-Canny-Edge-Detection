"""Comparison methods. OpenCV Canny and the gradient filters are baselines only.

Every function takes a float grayscale array and returns a uint8 edge map (0 or
255) of the same size. Sobel, Prewitt and Laplacian give a gradient or second
derivative response, so a threshold is needed to get a binary map. They use
Otsu's method on the response and do not thin edges, which makes their edges
thicker than Canny's. That difference is part of what the comparison shows.
"""
from __future__ import annotations

import contextlib
import io
import os
import sys
import tempfile
import warnings

import numpy as np
import PIL.Image
import scipy.ndimage as ndi

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _blur(gray, sigma):
    return ndi.gaussian_filter(gray, sigma, mode="reflect") if sigma > 0 else gray


def _binary_from_response(resp):
    from skimage.filters import threshold_otsu

    if resp.max() <= 0:
        return np.zeros(resp.shape, np.uint8)
    thr = threshold_otsu(resp)
    return ((resp > thr) * 255).astype(np.uint8)


def opencv_canny(gray, low=20, high=40, blur_sigma=1.0):
    import cv2

    u8 = np.clip(np.rint(_blur(gray, blur_sigma)), 0, 255).astype(np.uint8)
    return cv2.Canny(u8, low, high, L2gradient=True)


def sobel_edges(gray, blur_sigma=1.0):
    import cv2

    g = _blur(gray, blur_sigma).astype(np.float32)
    mag = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3))
    return _binary_from_response(mag)


def prewitt_edges(gray, blur_sigma=1.0):
    g = _blur(gray, blur_sigma)
    kx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=np.float64)
    mag = np.hypot(ndi.correlate(g, kx, mode="reflect"), ndi.correlate(g, kx.T, mode="reflect"))
    return _binary_from_response(mag)


def laplacian_edges(gray, blur_sigma=1.0):
    import cv2

    g = _blur(gray, blur_sigma).astype(np.float32)
    return _binary_from_response(np.abs(cv2.Laplacian(g, cv2.CV_32F, ksize=3)))


def original_manual_canny(gray_or_rgb, sigma=1.0, low=20, high=40):
    """Run the unchanged original loop implementation (canny_original.py).

    The original works on a file, prints progress and returns an array that is
    smaller than the input, with garbage in its first rows and columns. Here the
    result is placed on a canvas of the input size (the original's index (y, x)
    refers to input pixel (y, x)), so it can be compared. It is slow, so use it on small images.
    """
    sys.path.insert(0, ROOT)
    import canny_original

    arr = np.asarray(gray_or_rgb)
    img = PIL.Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "in.png")
        img.save(path)
        with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = canny_original.cannyEdgeDetector(path, sigma, low, high)
    canvas = np.zeros(arr.shape[:2], np.uint8)
    canvas[: out.shape[0], : out.shape[1]] = (out > 0) * 255
    return canvas

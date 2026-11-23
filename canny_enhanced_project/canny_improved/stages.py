"""The six Canny stages, vectorised with NumPy/SciPy.

The maths follows the original canny.py: same Gaussian kernel, Sobel kernels,
five direction bins and 255 / 75 / 0 threshold codes. Every place where the
behaviour deliberately differs from the original is marked FIX.

Unlike the original, all stage outputs keep the size of the input image. Borders
are handled by reflecting the image, so no rows or columns are lost.
"""
from __future__ import annotations

import math

import numpy as np
import PIL.Image
import scipy.ndimage as ndi
import scipy.stats as st

K = 3          # Gaussian half width (7 x 7 kernel), as in the original
STRONG = 255
WEAK = 75

SOBEL_X = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float64)
SOBEL_Y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)


def to_gray(image) -> np.ndarray:
    """Return float64 grayscale (0 to 255) from a path, PIL image or array.

    Colour uses the original weights 0.2126 R + 0.7152 G + 0.0722 B.
    """
    if isinstance(image, str):
        image = PIL.Image.open(image)
    if isinstance(image, PIL.Image.Image):
        image = np.asarray(image if image.mode == "L" else image.convert("RGB"))
    a = np.asarray(image, dtype=np.float64)
    if a.ndim == 3:
        a = a[..., :3]
        return 0.2126 * a[..., 0] + 0.7152 * a[..., 1] + 0.0722 * a[..., 2]
    return a


def gaussian_kernel(sigma: float, k: int = K) -> np.ndarray:
    """Same kernel as the original gkern(2K+1, sigma), including its square-root step."""
    n = 2 * k + 1
    interval = (2 * sigma + 1.0) / n
    x = np.linspace(-sigma - interval / 2.0, sigma + interval / 2.0, n + 1)
    kern1d = np.diff(st.norm.cdf(x))
    raw = np.sqrt(np.outer(kern1d, kern1d))
    return raw / raw.sum()


def gaussian_kernel_1d(sigma: float) -> np.ndarray:
    """True 1-D Gaussian with standard deviation sigma (in pixels), radius ceil(3 sigma)."""
    r = max(1, int(math.ceil(3 * sigma)))
    x = np.arange(-r, r + 1, dtype=np.float64)
    k = np.exp(-(x * x) / (2.0 * sigma * sigma))
    return k / k.sum()


def smooth_gaussian(gray: np.ndarray, sigma: float, kernel: str = "standard") -> np.ndarray:
    """S1. Gaussian smoothing, same size as the input, borders reflected.

    kernel="standard": a true Gaussian whose standard deviation is sigma pixels
    (separable, so it is also faster). This is the improved default.
    kernel="original": the original 7 x 7 kernel. FIX note: in the original,
    sigma is the number of standard deviations the 7-tap window covers, so
    sigma = 1 gives an almost flat, box-like kernel and a larger sigma gives
    LESS smoothing. It is kept only to reproduce the original results.

    FIX: the original filter() also wrote results at the wrong offset and left
    the first K rows and columns uninitialised (numpy.empty).
    """
    if kernel == "original":
        return ndi.correlate(gray, gaussian_kernel(sigma), mode="reflect")
    if kernel != "standard":
        raise ValueError("kernel must be 'standard' or 'original'")
    k1 = gaussian_kernel_1d(sigma)
    return ndi.correlate1d(ndi.correlate1d(gray, k1, axis=0, mode="reflect"), k1, axis=1, mode="reflect")


def quantize(a: np.ndarray, mode: str = "round") -> np.ndarray:
    """Turn smoothed values into whole grey levels (the original truncated)."""
    if mode == "round":
        return np.rint(a)
    if mode == "trunc":
        return np.trunc(a)
    if mode == "none":
        return a
    raise ValueError("quantize must be 'round', 'trunc' or 'none'")


def sobel(smooth: np.ndarray):
    """S2. Horizontal and vertical Sobel gradients, same size as the input."""
    return (ndi.correlate(smooth, SOBEL_X, mode="reflect"),
            ndi.correlate(smooth, SOBEL_Y, mode="reflect"))


def magnitude_direction(sx: np.ndarray, sy: np.ndarray):
    """S3. Gradient magnitude and the original five-bin direction code (0 to 4)."""
    mag = np.sqrt(sx * sx + sy * sy)
    safe = np.where(sx == 0, 1.0, sx)
    angle = np.arctan(sy / safe)
    angle = np.where(sx == 0, np.where(sy != 0, math.pi / 2, 0.0), angle)
    direction = (np.round(angle * (5.0 / math.pi)) + 5) % 5
    return mag, direction


def non_max_suppression(mag: np.ndarray, direction: np.ndarray, ties: str = "keep_one") -> np.ndarray:
    """S4. Keep a pixel only if it beats both neighbours along its gradient direction.

    ties: "suppress_all" is the original rule (a pixel equal to either neighbour is
    removed, so a perfectly symmetric edge disappears). "keep_one" removes a pixel
    only if it is smaller than the earlier neighbour or not larger than the later
    one, so exactly one pixel of an equal pair survives.

    FIX: the original compared against a magnitude array it was modifying while
    looping, so the result depended on loop order. Here every pixel is compared
    with the unchanged magnitudes. Ties are suppressed and the 1-pixel border is
    zero, as in the original.
    """
    d = (direction % 4)[1:-1, 1:-1]
    c = mag[1:-1, 1:-1]
    a = np.where(d == 0, mag[1:-1, :-2], np.where(d == 1, mag[:-2, 2:],
                 np.where(d == 2, mag[:-2, 1:-1], mag[:-2, :-2])))
    b = np.where(d == 0, mag[1:-1, 2:], np.where(d == 1, mag[2:, :-2],
                 np.where(d == 2, mag[2:, 1:-1], mag[2:, 2:])))
    out = np.zeros_like(mag)
    if ties not in ("keep_one", "suppress_all"):
        raise ValueError("ties must be 'keep_one' or 'suppress_all'")
    drop = ((c <= a) | (c <= b)) if ties == "suppress_all" else ((c < a) | (c <= b))
    out[1:-1, 1:-1] = np.where(drop, 0.0, c)
    return out


def double_threshold(img: np.ndarray, low: float, high: float) -> np.ndarray:
    """S5. Strong (255), weak (75) or rejected (0).

    FIX: all masks come from the same input. The original overwrote values while
    comparing them, which broke for low > 75 or high >= 255.
    """
    out = np.zeros(img.shape, dtype=np.uint8)
    out[(img >= low) & (img <= high)] = WEAK
    out[img > high] = STRONG
    return out


def hysteresis(img: np.ndarray, mode: str = "single_pass") -> np.ndarray:
    """S6. Decide which weak pixels become edges.

    single_pass: the original rule. Scan in row order and keep a weak pixel when
    one of six neighbours is strong, counting weak pixels promoted earlier in the
    scan. Only weak pixels are visited, so it is fast.
    full: standard hysteresis. Keep weak pixels connected (8-neighbourhood) to a
    strong pixel through other weak or strong pixels.
    """
    if mode == "single_pass":
        p = np.pad(img, 1)
        ys, xs = np.nonzero(p == WEAK)
        for i, j in zip(ys.tolist(), xs.tolist()):
            hit = (p[i + 1, j] == STRONG or p[i - 1, j] == STRONG or p[i, j + 1] == STRONG
                   or p[i, j - 1] == STRONG or p[i + 1, j + 1] == STRONG or p[i - 1, j - 1] == STRONG)
            p[i, j] = STRONG if hit else 0
        return p[1:-1, 1:-1].copy()
    if mode == "full":
        cand = img > 0
        labels, _ = ndi.label(cand, structure=np.ones((3, 3), dtype=bool))
        keep_ids = np.unique(labels[img == STRONG])
        keep_ids = keep_ids[keep_ids > 0]
        out = np.zeros_like(img)
        out[np.isin(labels, keep_ids) & cand] = STRONG
        return out
    raise ValueError("hysteresis must be 'single_pass' or 'full'")

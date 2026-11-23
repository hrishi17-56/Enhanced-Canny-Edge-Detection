"""Slow, plain-loop reference versions of each stage, used only to check the fast code.

They follow the original canny.py closely, with the same fixes as the fast code
(correct valid-region filtering, suppression against an unchanged copy).
"""
import math
from collections import deque

import numpy as np

STRONG, WEAK = 255, 75


def ref_correlate_valid(kernel, img, k):
    h, w = img.shape
    out = np.zeros((h - 2 * k, w - 2 * k))
    for y in range(k, h - k):
        for x in range(k, w - k):
            total = 0.0
            for dy in range(-k, k + 1):
                for dx in range(-k, k + 1):
                    total += kernel[dy + k][dx + k] * img[y + dy][x + dx]
            out[y - k][x - k] = total
    return out


def ref_quantize(a, mode):
    if mode == "round":
        return np.rint(a)
    if mode == "trunc":
        return np.trunc(a)
    return a


def ref_magnitude_direction(sx, sy):
    h, w = sx.shape
    mag = np.zeros((h, w))
    deg = np.zeros((h, w))
    for y in range(h):
        for x in range(w):
            mag[y][x] = math.sqrt(sx[y][x] * sx[y][x] + sy[y][x] * sy[y][x])
            if sx[y][x] == 0:
                deg[y][x] = math.pi / 2 if sy[y][x] != 0 else 0
            else:
                deg[y][x] = math.atan(sy[y][x] / sx[y][x])
    deg = (np.round(deg * (5.0 / math.pi)) + 5) % 5
    return mag, deg


def ref_nms(mag, deg, ties="keep_one"):
    h, w = mag.shape
    out = np.zeros((h, w))
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            d = deg[y][x] % 4
            c = mag[y][x]
            if d == 0:
                a, b = mag[y][x - 1], mag[y][x + 1]
            elif d == 1:
                a, b = mag[y - 1][x + 1], mag[y + 1][x - 1]
            elif d == 2:
                a, b = mag[y - 1][x], mag[y + 1][x]
            else:
                a, b = mag[y - 1][x - 1], mag[y + 1][x + 1]
            drop = (c <= a or c <= b) if ties == "suppress_all" else (c < a or c <= b)
            out[y][x] = 0 if drop else c
    return out


def ref_threshold(img, low, high):
    out = np.zeros(img.shape, dtype=np.uint8)
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            v = img[y][x]
            if v > high:
                out[y][x] = STRONG
            elif v >= low:
                out[y][x] = WEAK
    return out


def ref_hysteresis_single_pass(img):
    p = np.pad(img, 1).copy()
    h, w = p.shape
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            if p[i][j] == WEAK:
                if (p[i + 1][j] == STRONG or p[i - 1][j] == STRONG or p[i][j + 1] == STRONG
                        or p[i][j - 1] == STRONG or p[i + 1][j + 1] == STRONG or p[i - 1][j - 1] == STRONG):
                    p[i][j] = STRONG
                else:
                    p[i][j] = 0
    return p[1:-1, 1:-1]


def ref_hysteresis_full(img):
    h, w = img.shape
    out = np.zeros_like(img)
    seen = np.zeros((h, w), dtype=bool)
    queue = deque()
    for y in range(h):
        for x in range(w):
            if img[y][x] == STRONG:
                out[y][x] = STRONG
                seen[y][x] = True
                queue.append((y, x))
    while queue:
        y, x = queue.popleft()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and not seen[ny][nx] and img[ny][nx] == WEAK:
                    seen[ny][nx] = True
                    out[ny][nx] = STRONG
                    queue.append((ny, nx))
    return out

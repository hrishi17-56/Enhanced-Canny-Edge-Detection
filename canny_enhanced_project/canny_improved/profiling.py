"""Timing and memory helpers."""
from __future__ import annotations

import time
import tracemalloc

import numpy as np


def time_call(fn, repeats=5, warmup=1):
    """Median and minimum wall-clock time of fn() in milliseconds."""
    for _ in range(warmup):
        fn()
    runs = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        runs.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(runs)), float(np.min(runs))


def peak_memory_mb(fn):
    """Peak memory allocated during fn() in MB, as seen by tracemalloc.

    NumPy arrays are tracked. Memory used inside OpenCV's C++ code is not, so
    treat the number as an estimate for the Python/NumPy part only.
    """
    tracemalloc.start()
    tracemalloc.reset_peak()
    fn()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / (1024 * 1024)

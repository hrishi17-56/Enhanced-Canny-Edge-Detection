"""Plots: stage-by-stage grid and side-by-side method comparison."""
from __future__ import annotations

import numpy as np

STAGE_TITLES = ["Original", "Grayscale", "Smoothed", "Gradient magnitude",
                "Non-maximum suppressed", "Thresholded", "Final edges"]


def _show(ax, img, title, cmap="gray"):
    """Show one panel. Gradient panels are stretched to their 99.5th percentile so faint edges stay visible."""
    if img.ndim == 3:
        ax.imshow(img)
    elif title in ("Gradient magnitude", "Non-maximum suppressed"):
        ax.imshow(img, cmap=cmap, vmin=0, vmax=max(float(np.percentile(img, 99.5)), 1e-9))
    else:
        ax.imshow(img, cmap=cmap, vmin=0, vmax=255)
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def stage_grid(original, result, path=None, title=None):
    """Seven panels: the input and the output of each stage (needs keep_stages=True)."""
    import matplotlib.pyplot as plt

    s = result.stages
    imgs = [original, s["gray"], s["smoothed"], s["magnitude"], s["nms"], s["thresholded"], result.edges]
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    for ax, im, name in zip(axes.ravel(), imgs, STAGE_TITLES):
        _show(ax, np.asarray(im), name)
    axes.ravel()[-1].axis("off")
    axes.ravel()[-1].text(0.05, 0.5, f"low = {result.thresholds[0]:.1f}\nhigh = {result.thresholds[1]:.1f}\n"
                          f"total = {result.timings['total']:.1f} ms", fontsize=11, va="center")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=110)
    return fig


def comparison_grid(original, maps: dict, path=None, title=None, ncols=4):
    """Original image plus one panel per method."""
    import matplotlib.pyplot as plt

    names = ["Input"] + list(maps)
    panels = [original] + list(maps.values())
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 3.4 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, im, name in zip(axes, panels, names):
        _show(ax, np.asarray(im), name)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=110)
    return fig

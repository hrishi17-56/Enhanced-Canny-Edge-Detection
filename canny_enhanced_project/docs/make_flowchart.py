"""Draws docs/flowchart.png (pipeline and project architecture)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(figsize=(13, 7.2))
ax.set_xlim(0, 13); ax.set_ylim(0, 7.2); ax.axis("off")


def box(x, y, w, h, text, fc="#e8f0fe", ec="#3b5bdb", fs=9, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.08", fc=fc, ec=ec, lw=1.3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal")


def arrow(x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", lw=1.3, color="#444"))


ax.text(6.5, 6.95, "Improved Canny: pipeline and project architecture", ha="center", fontsize=13, fontweight="bold")
# main pipeline (top row and second row)
row1 = [("Input image\n(path, upload\nor array)", "#fff3bf", "#e67700"), ("1. Grayscale\n0.2126R+0.7152G\n+0.0722B", None, None),
        ("2. Noise reduction\nGaussian | Median |\nBilateral | NLM", "#d3f9d8", "#2b8a3e"), ("3. Sobel\ngradients\nGx, Gy", None, None),
        ("4. Magnitude and\ndirection", None, None)]
x = 0.3
for i, (t, fc, ec) in enumerate(row1):
    box(x, 5.2, 2.2, 1.2, t, fc or "#e8f0fe", ec or "#3b5bdb", bold=False)
    if i:
        arrow(x - 0.25, 5.8, x, 5.8)
    x += 2.5
arrow(12.3, 5.8, 12.3, 5.35) if False else None
row2 = [("5. Non-maximum\nsuppression\n(ties: keep one)", None, None), ("6. Threshold selection\nFixed | Median rule |\nOtsu", "#d3f9d8", "#2b8a3e"),
        ("7. Double\nthresholding\n255 / 75 / 0", None, None), ("8. Hysteresis\nsingle pass | full", None, None),
        ("Final edge map\n(uint8, 0 or 255)", "#ffe3e3", "#c92a2a")]
x = 0.3
for i, (t, fc, ec) in enumerate(row2):
    box(x, 3.4, 2.2, 1.2, t, fc or "#e8f0fe", ec or "#3b5bdb")
    if i:
        arrow(x - 0.25, 4.0, x, 4.0)
    x += 2.5
arrow(11.1, 5.2, 1.4, 4.6)   # link row 1 end to row 2 start
# lower layer
box(0.3, 1.2, 3.6, 1.5, "Baselines (comparison only)\nOriginal loop script | OpenCV Canny\nSobel | Prewitt | Laplacian", "#f3d9fa", "#862e9c")
box(4.3, 1.2, 4.0, 1.5, "Evaluation (evaluate.py)\nTime per stage, resolution, memory\nPrecision, recall, F1 (drawn scenes)\nAgreement with OpenCV (photos)", "#fff3bf", "#e67700")
box(8.7, 1.2, 4.0, 1.5, "Outputs\nStage grids, side-by-side comparisons\nCSV tables, summary.md\nStreamlit app, Colab notebook", "#e8f0fe", "#3b5bdb")
arrow(11.4, 3.4, 11.4, 2.75); arrow(2.1, 3.4, 2.1, 2.75); arrow(3.95, 1.95, 4.25, 1.95); arrow(8.35, 1.95, 8.65, 1.95)
ax.text(6.5, 0.55, "Green boxes are the configurable parts. Fixes to the original: border handling, in-place suppression, tie rule, Gaussian kernel, threshold overwrite.",
        ha="center", fontsize=8.5, color="#333")
fig.savefig("docs/flowchart.png", dpi=130, bbox_inches="tight")

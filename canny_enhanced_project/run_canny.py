"""Command-line use of the improved Canny detector.

Examples
    python run_canny.py "images/bird 2.jpg"
    python run_canny.py images/Lenna.png --denoise bilateral --threshold otsu --stages
    python run_canny.py my_photo.jpg --threshold fixed --low 30 --high 80 --out output/my_photo_edges.png
"""
import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402
import PIL.Image  # noqa: E402

from canny_improved import canny_improved  # noqa: E402
from canny_improved.visualize import stage_grid  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description="Improved manual Canny edge detection")
    ap.add_argument("image")
    ap.add_argument("--denoise", default="gaussian", choices=["gaussian", "median", "bilateral", "nlm", "none"])
    ap.add_argument("--sigma", type=float, default=1.0, help="Gaussian standard deviation in pixels")
    ap.add_argument("--threshold", default="fixed", choices=["fixed", "median", "otsu"])
    ap.add_argument("--low", type=float, default=20)
    ap.add_argument("--high", type=float, default=40)
    ap.add_argument("--hysteresis", default="single_pass", choices=["single_pass", "full"])
    ap.add_argument("--out", help="output file (default output/<name>_edges.png)")
    ap.add_argument("--stages", action="store_true", help="also save a stage-by-stage figure")
    a = ap.parse_args(argv)
    if not os.path.exists(a.image):
        sys.exit(f"Input image not found: {a.image}")
    img = PIL.Image.open(a.image)
    rgb = np.asarray(img if img.mode == "L" else img.convert("RGB"))
    try:
        res = canny_improved(rgb, a.denoise, a.sigma, None, a.threshold, a.low, a.high, a.hysteresis, keep_stages=a.stages)
    except ValueError as exc:
        sys.exit(f"Error: {exc}")
    stem = os.path.splitext(os.path.basename(a.image))[0].replace(" ", "_")
    out = a.out or os.path.join("output", f"{stem}_edges.png")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    PIL.Image.fromarray(res.edges).save(out)
    print(f"Saved {out}: {res.edges.shape[1]}x{res.edges.shape[0]}, thresholds {res.thresholds[0]:.1f}/{res.thresholds[1]:.1f}, "
          f"{res.timings['total']:.1f} ms")
    if a.stages:
        sg = out.replace(".png", "_stages.png")
        stage_grid(rgb, res, sg, title=os.path.basename(a.image))
        print(f"Saved {sg}")


if __name__ == "__main__":
    main()

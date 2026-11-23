"""Streamlit interface. Run with:  streamlit run app.py"""
import io
import os
import time

import numpy as np
import PIL.Image
import streamlit as st

from canny_improved import baselines as B
from canny_improved import canny_improved
from canny_improved.metrics import agreement_f1, edge_density
from canny_improved.stages import to_gray

ROOT = os.path.dirname(os.path.abspath(__file__))
SAMPLES = sorted(f for f in os.listdir(os.path.join(ROOT, "images")) if f.lower().endswith((".jpg", ".jpeg", ".png")))
ALGORITHMS = ["Improved Canny (mine)", "Original manual Canny", "OpenCV Canny", "Sobel", "Prewitt", "Laplacian"]

st.set_page_config(page_title="Canny edge detection", layout="wide")
st.title("Implementation and enhancement of Canny edge detection")

with st.sidebar:
    st.header("Image")
    upload = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "bmp"])
    sample = st.selectbox("or choose a sample", SAMPLES, index=SAMPLES.index("bird 2.jpg") if "bird 2.jpg" in SAMPLES else 0)
    max_side = st.slider("Shrink so the longest side is at most (px)", 128, 1600, 640, 32)
    st.header("Method")
    algo = st.selectbox("Algorithm", ALGORITHMS)
    den = st.selectbox("Noise reduction", ["gaussian", "median", "bilateral", "nlm", "none"])
    sigma = st.slider("Gaussian sigma (px)", 0.5, 4.0, 1.0, 0.1)
    thr = st.selectbox("Threshold selection", ["fixed", "median", "otsu"])
    low = st.slider("Low threshold (fixed mode)", 1, 300, 20)
    high = st.slider("High threshold (fixed mode)", 2, 600, 40)
    hyst = st.selectbox("Hysteresis", ["single_pass", "full"])
    compare = st.checkbox("Compare with all other algorithms", value=False)

path = upload if upload is not None else os.path.join(ROOT, "images", sample)
img = PIL.Image.open(path)
img = img.convert("L") if img.mode in ("L", "LA", "P") and img.mode == "L" else img.convert("RGB")
if max(img.size) > max_side:
    s = max_side / max(img.size)
    img = img.resize((max(16, round(img.width * s)), max(16, round(img.height * s))))
rgb = np.asarray(img)
gray = to_gray(rgb)


def run(name):
    t0 = time.perf_counter()
    if name == "Improved Canny (mine)":
        out = canny_improved(rgb, denoise=den, sigma=sigma, threshold=thr, low=low, high=high, hysteresis=hyst).edges
    elif name == "Original manual Canny":
        small = gray if max(gray.shape) <= 256 else np.asarray(PIL.Image.fromarray(gray.astype(np.uint8)).resize(
            (max(16, round(gray.shape[1] * 256 / max(gray.shape))), max(16, round(gray.shape[0] * 256 / max(gray.shape))))), dtype=np.float64)
        out = B.original_manual_canny(small, sigma, low, high)
        out = np.asarray(PIL.Image.fromarray(out).resize((gray.shape[1], gray.shape[0]), PIL.Image.NEAREST))
    elif name == "OpenCV Canny":
        out = B.opencv_canny(gray, low, high, sigma)
    elif name == "Sobel":
        out = B.sobel_edges(gray, sigma)
    elif name == "Prewitt":
        out = B.prewitt_edges(gray, sigma)
    else:
        out = B.laplacian_edges(gray, sigma)
    return out, (time.perf_counter() - t0) * 1000.0


names = ALGORITHMS if compare else [algo]
if "Original manual Canny" in names:
    st.info("The original loop implementation is slow, so it runs on a copy shrunk to 256 px and is scaled back up for display.")
results = {n: run(n) for n in names}

c0, *cols = st.columns(1 + min(len(names), 3))
c0.image(rgb, caption=f"Input ({gray.shape[1]} x {gray.shape[0]})", use_container_width=True)
for i, (n, (edges, ms)) in enumerate(results.items()):
    col = cols[i % len(cols)]
    col.image(edges, caption=f"{n}: {ms:.1f} ms, edge density {100 * edge_density(edges):.1f}%", use_container_width=True)

if compare and len(results) > 1:
    ref_edges = results["OpenCV Canny"][0]
    st.subheader("Agreement with OpenCV Canny")
    st.caption("This is agreement between two methods, not accuracy. No ground-truth edges exist for uploaded photos.")
    st.table({n: [f"{agreement_f1(e, ref_edges):.3f}", f"{ms:.1f}"] for n, (e, ms) in results.items()}
             | {"": ["agreement F1", "time (ms)"]})

main_edges = results[names[0]][0]
buf = io.BytesIO()
PIL.Image.fromarray(main_edges).save(buf, format="PNG")
st.download_button(f"Download {names[0]} result (PNG)", buf.getvalue(), file_name="edges.png", mime="image/png")

if algo == "Improved Canny (mine)" and not compare:
    with st.expander("Show every stage"):
        r = canny_improved(rgb, denoise=den, sigma=sigma, threshold=thr, low=low, high=high, hysteresis=hyst, keep_stages=True)
        a, b, c, d = st.columns(4)
        norm = lambda x: np.clip(x / max(x.max(), 1e-9) * 255, 0, 255).astype(np.uint8)  # noqa: E731
        a.image(norm(r.stages["gray"]), caption="Grayscale")
        b.image(norm(r.stages["smoothed"]), caption="Smoothed")
        c.image(norm(r.stages["magnitude"]), caption="Gradient magnitude")
        d.image(norm(r.stages["nms"]), caption="Non-maximum suppressed")
        e, f, g = st.columns(3)
        e.image(r.stages["thresholded"], caption=f"Thresholded (low {r.thresholds[0]:.1f}, high {r.thresholds[1]:.1f})")
        f.image(r.edges, caption="Final edges")
        g.json({k: round(v, 2) for k, v in r.timings.items()})

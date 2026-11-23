# Implementation and enhancement of Canny edge detection

A manual, six-stage Canny edge detector written with NumPy and SciPy, plus noise-reduction and threshold options, comparison against other edge detectors, timing and accuracy experiments, a Colab notebook and a small Streamlit app. It runs on the CPU only, so a Windows laptop with integrated graphics is enough and Google Colab needs no GPU.

The base code is the Canny implementation from the GitHub project `hasbisevinc/Canny-Edge-Detection-Algorithm`. The download had no licence file, so check the upstream licence before publishing this project. The original script is kept unchanged in `canny_original.py` as the baseline.

OpenCV's Canny is used only as a comparison. The detector itself is not built on it.

## What changed compared with the original

| Area | Original script | Improved version |
|---|---|---|
| Border handling | `filter()` leaves the first 3 rows and columns uninitialised (`numpy.empty`) and shifts the result, which draws a false line along the top and left edge | Reflected borders, every pixel computed, output has the input size |
| Gaussian kernel | `sigma` is the number of standard deviations the 7x7 window covers, so `sigma=1` gives an almost flat kernel and a larger `sigma` smooths less | True Gaussian with a standard deviation in pixels. The original kernel stays available as `kernel="original"` |
| Non-maximum suppression | Modifies the magnitude array while looping, so the result depends on loop order. Equal neighbours are both removed, so a perfectly sharp edge can vanish | Compares against unchanged magnitudes. One pixel of an equal pair is kept (`nms_ties="keep_one"`) |
| Thresholding | Overwrites values while comparing them, which breaks for `low > 75` or `high >= 255` | All masks come from the same input |
| Speed | Python loops over every pixel | Vectorised NumPy. Hysteresis visits only weak pixels |
| Threshold choice | Two numbers typed in the script | `fixed`, `median` rule or `otsu` |
| Noise reduction | One Gaussian | Gaussian, median, bilateral (written in NumPy) and non-local means (OpenCV) |
| Interface | Edit the script, output name fixed to `lenna.png` | Python API, command line, Streamlit app, notebook |

The mathematical steps are unchanged: grayscale weights, Sobel kernels, five direction bins, double threshold codes 255 / 75 / 0 and the single-pass hysteresis rule. A test replays the original in-place suppression on the corrected earlier stages and checks that it reproduces the original script's output.

## Folder layout

```
canny_original.py        original baseline (unchanged)
canny_improved/          improved implementation
    stages.py            the six stages
    denoise.py           Gaussian, median, bilateral, non-local means
    thresholds.py        fixed, median rule, Otsu
    pipeline.py          canny_improved(...) and its result object
    baselines.py         OpenCV Canny, Sobel, Prewitt, Laplacian, wrapper for the original
    metrics.py           precision, recall, F1, agreement
    synthetic.py         drawn scenes with known edges, noise, contrast and brightness changes
    profiling.py         timing and memory helpers
    visualize.py         stage grid and side-by-side comparison
app.py                   Streamlit interface
run_canny.py             command-line tool
evaluate.py              all experiments, writes results/
notebooks/Canny_Colab.ipynb
tests/                   43 automated tests
images/                  sample images
results/                 tables, figures and summary.md from the included run
docs/                    flowchart
```

## Install

Windows 11 (PowerShell, Python 3.10 or newer):

```
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If PowerShell blocks the activation script, run `Set-ExecutionPolicy -Scope Process Bypass` first. A GPU is not used.

Google Colab: open `notebooks/Canny_Colab.ipynb` in Colab (File > Upload notebook), run the cells from the top, and upload the project zip when cell 1 asks for it.

## Use

```python
from canny_improved import canny_improved

res = canny_improved("images/Lenna.png", denoise="bilateral", threshold="median", keep_stages=True)
res.edges        # uint8 map, 0 or 255
res.thresholds   # (low, high) that were used
res.timings      # milliseconds per step
res.stages       # gray, smoothed, magnitude, nms, thresholded
```

Command line:

```
python run_canny.py "images/bird 2.jpg" --threshold median --stages
python run_canny.py my_photo.jpg --denoise median --threshold otsu
```

Streamlit app (upload, method choice, threshold sliders, side-by-side view, timings, download):

```
streamlit run app.py
```

Tests and experiments:

```
python -m pytest -q tests
python evaluate.py --quick          # about a minute
python evaluate.py                  # full run, a few minutes
python evaluate.py --extra-images my_photos/
```

### Options of `canny_improved`

| Option | Values | Default |
|---|---|---|
| `denoise` | gaussian, median, bilateral, nlm, none | gaussian |
| `sigma` | Gaussian standard deviation in pixels | 1.0 |
| `threshold` | fixed (`low`, `high`), median rule, otsu | fixed (20, 40) |
| `hysteresis` | single_pass (original rule), full (connected components) | single_pass |
| `quantize` | round, trunc (original), none | round |
| `kernel` | standard, original | standard |
| `nms_ties` | keep_one, suppress_all (original) | keep_one |

To reproduce the original script's behaviour, use `kernel="original"`, `nms_ties="suppress_all"` and `quantize="trunc"`.

## Results from the included run

`results/summary.md` has every table. These were produced on a single-CPU Linux machine, not a laptop, so rerun `evaluate.py` on your own computer before quoting times.

- Speed: the improved version took 1.8 ms at 128x128 and 7.0 ms at 256x256, against 315 ms and 1350 ms for the original script (about 176 and 194 times faster). At 1024x1024 it took about 140 ms. OpenCV's compiled Canny was about 5 to 7 times faster than this NumPy version.
- Memory: peak NumPy memory is about 80 bytes per pixel (81 MB at 1024x1024). It is not lower than the loop version at small sizes, because vectorised code creates whole-image temporaries.
- Accuracy on drawn scenes with known edges (F1, 2 px tolerance): on clean scenes the improved version scored 0.969 and the original script 0.730. With noise of 20 grey levels, fixed thresholds fell to 0.219, while the median rule gave 0.946 and Otsu 0.855.
- The median rule is not safe everywhere. At a quarter of the normal contrast it dropped to 0.201, while fixed thresholds scored 0.886 and Otsu 0.850.
- On the six photographs there is no ground truth, so only agreement with OpenCV Canny is reported.

## Metrics and their limits

Precision, recall and F1 compare detected edge pixels with known edge pixels within a 2 px tolerance. They can only be computed for the drawn scenes, because the project has no ground-truth edge maps for photographs. The match is not one-to-one, and drawn scenes are simpler than photographs. Agreement with OpenCV Canny measures similarity to another algorithm, not correctness. Execution time depends on the machine. Memory comes from `tracemalloc` and does not include memory used inside OpenCV.

## Known limitations

- Bilateral and non-local means use default parameters that were not tuned. At noise 25 they left a lot of noise in the edge map.
- Non-local means calls OpenCV. The other denoisers are written in NumPy or SciPy.
- Sobel, Prewitt and Laplacian use an Otsu threshold and do not thin edges, so they produce thick edges.
- The Original manual Canny option in the app shrinks the image to 256 px because the loop script is slow.
- The median and Otsu threshold rules are heuristics.

## Architecture

![Flowchart](docs/flowchart.png)

## Future work

- Test on a public edge dataset with ground truth, such as BSDS500, and use one-to-one matching.
- Tune the bilateral and non-local means parameters per noise level.
- Use float32 and in-place operations to cut memory.
- Add adaptive local thresholds and a noise estimate to choose the filter automatically.
- Compile hysteresis with Numba for very large images.

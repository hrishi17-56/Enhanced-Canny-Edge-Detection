import nbformat as nbf

nb = nbf.v4.new_notebook()
md = lambda t: nb.cells.append(nbf.v4.new_markdown_cell(t))  # noqa: E731
code = lambda t: nb.cells.append(nbf.v4.new_code_cell(t))    # noqa: E731

md("""# Implementation and enhancement of Canny edge detection

This notebook runs the improved manual Canny project in Google Colab. A GPU is not needed, so the default CPU runtime is fine.

**Before you start:** zip the project folder (or use `canny_enhanced_project.zip`). Run the cells from top to bottom. Cell 1 asks you to upload the zip.""")

code('''# Cell 1: get the project into Colab (upload the zip) and move into its folder
import glob, os, sys, zipfile
IN_COLAB = "google.colab" in sys.modules
if IN_COLAB and not os.path.exists("canny_improved"):
    from google.colab import files
    uploaded = files.upload()                      # choose the project zip in the dialog
    zip_name = next(iter(uploaded))
    zipfile.ZipFile(zip_name).extractall(".")
    hit = glob.glob("**/canny_improved/__init__.py", recursive=True)[0]
    os.chdir(os.path.dirname(os.path.dirname(hit)) or ".")
sys.path.insert(0, os.getcwd())
print("Working folder:", os.getcwd())
print(sorted(os.listdir(".")))''')

code('''# Cell 2: install the requirements (most are already present in Colab)
import subprocess
r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], capture_output=True, text=True)
print(r.stdout[-500:] or "requirements ready", r.stderr[-300:] if r.returncode else "")''')

code('''# Cell 3: run the automated tests
r = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests"], capture_output=True, text=True)
print(r.stdout[-1200:])''')

md("## 1. Audit: original script against the improved version\nThe original loop script is slow, so a small image is used.")
code('''import time
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
from canny_improved import canny_improved, baselines as B
from canny_improved.stages import to_gray

IMAGE = "images/bird 2.jpg"                 # change this to any image path
rgb = np.asarray(PIL.Image.open(IMAGE).convert("RGB"))
small = np.asarray(PIL.Image.fromarray(rgb).resize((200, round(200 * rgb.shape[0] / rgb.shape[1]))))
t0 = time.perf_counter(); orig = B.original_manual_canny(small); t_orig = time.perf_counter() - t0
t0 = time.perf_counter(); new = canny_improved(small, kernel="original", nms_ties="suppress_all", quantize="trunc").edges; t_new = time.perf_counter() - t0
best = canny_improved(small)
print(f"original script {t_orig*1000:.0f} ms | improved {best.timings['total']:.1f} ms")
fig, ax = plt.subplots(1, 3, figsize=(13, 4))
for a, im, t in zip(ax, [orig, new, best.edges], ["Original script (note the border line)", "Improved, original-compatible settings", "Improved, default settings"]):
    a.imshow(im, cmap="gray"); a.set_title(t, fontsize=9); a.axis("off")
plt.show()''')

md("## 2. Every stage, side by side")
code('''from canny_improved.visualize import stage_grid
res = canny_improved(rgb, threshold="median", keep_stages=True)
stage_grid(rgb, res, title=IMAGE)
plt.show()''')

md("## 3. Compare noise reduction methods and threshold choices\nThe image gets Gaussian noise first, so the difference between the filters shows.")
code('''from canny_improved.synthetic import add_noise
from canny_improved.visualize import comparison_grid
noisy = add_noise(to_gray(small), 20, 0)
maps = {f"{d} + {t}": canny_improved(noisy, denoise=d, threshold=t).edges
        for d in ["gaussian", "median", "bilateral", "nlm"] for t in ["fixed", "median", "otsu"]}
comparison_grid(noisy, maps, title="Noise sigma 20", ncols=5)
plt.show()''')

md("## 4. Compare with OpenCV Canny, Sobel, Prewitt and Laplacian\nOpenCV Canny is only a baseline here.")
code('''g = to_gray(rgb)
from canny_improved.thresholds import select_thresholds
maps = {"Improved (median thr)": canny_improved(rgb, threshold="median").edges,
        "Improved (Otsu thr)": canny_improved(rgb, threshold="otsu").edges,
        "OpenCV Canny": B.opencv_canny(g, *select_thresholds("median", g, g)),
        "Sobel": B.sobel_edges(g), "Prewitt": B.prewitt_edges(g), "Laplacian": B.laplacian_edges(g)}
comparison_grid(rgb, maps, title=IMAGE)
plt.show()''')

md("## 5. Try your own image\nIn Colab this opens an upload dialog. Elsewhere it reuses the sample image.")
code('''if IN_COLAB:
    from google.colab import files
    up = files.upload()
    if up:
        name = next(iter(up)); rgb = np.asarray(PIL.Image.open(name).convert("RGB")); IMAGE = name
res = canny_improved(rgb, denoise="gaussian", threshold="median", keep_stages=True)
print("thresholds used:", [round(x, 1) for x in res.thresholds], "| total ms:", round(res.timings["total"], 1))
stage_grid(rgb, res, title=IMAGE); plt.show()''')

md("## 6. Performance evaluation\nThe quick run takes about a minute. The full run (`python evaluate.py`) takes a few minutes and uses more scenes and sizes.")
code('''r = subprocess.run([sys.executable, "evaluate.py", "--quick", "--outdir", "results_colab"], capture_output=True, text=True)
print(r.stderr[-500:] if r.returncode else "done")
print(open("results_colab/summary.md").read()[:3500])''')
code('''for f in ["f1_vs_noise.png", "speed_vs_resolution.png"]:
    plt.figure(figsize=(7, 4.5)); plt.imshow(PIL.Image.open(f"results_colab/figures/{f}")); plt.axis("off"); plt.show()''')

md("""## 7. Save your results (optional)
Colab erases its files when the session ends. Run this cell to keep the results in Google Drive.""")
code('''# Uncomment to copy results to Drive
# from google.colab import drive; drive.mount("/content/drive")
# import shutil; shutil.copytree("results_colab", "/content/drive/MyDrive/canny_results", dirs_exist_ok=True)''')

md("""## Streamlit interface
The Streamlit app (`app.py`) is meant for a laptop: `streamlit run app.py`. Running it inside Colab needs a tunnelling service, so it is not set up here.""")
nbf.write(nb, "notebooks/Canny_Colab.ipynb")

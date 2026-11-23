import os
import sys

import numpy as np
import pytest
import scipy.ndimage as ndi

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [ROOT, HERE]

import reference_loops as ref  # noqa: E402
from canny_improved import canny_improved, to_gray  # noqa: E402
from canny_improved import baselines as B  # noqa: E402
from canny_improved import denoise as D  # noqa: E402
from canny_improved import stages as S  # noqa: E402
from canny_improved.metrics import edge_density, edge_prf  # noqa: E402
from canny_improved.synthetic import add_noise, change_brightness, change_contrast, resize, synthetic_scene  # noqa: E402
from canny_improved.thresholds import select_thresholds  # noqa: E402

IMG = lambda n: os.path.join(ROOT, "images", n)  # noqa: E731


def blocky(h=31, w=40, seed=0):
    rng = np.random.default_rng(seed)
    a = np.zeros((h, w))
    a[8:22, 10:30] = 180
    a[14:18, 15:25] = 60
    return np.clip(a + rng.normal(0, 6, a.shape), 0, 255)


# ---------- stages against plain-loop references (interior only; borders are reflected) ----------

def test_original_kernel_smoothing_matches_loop_in_interior():
    g = blocky()
    got = S.smooth_gaussian(g, 1.0, "original")
    want = ref.ref_correlate_valid(S.gaussian_kernel(1.0), g, S.K)
    assert got.shape == g.shape
    assert np.allclose(got[S.K:-S.K, S.K:-S.K], want, atol=1e-9)


def test_standard_gaussian_has_requested_std_and_matches_scipy():
    g = blocky(40, 40, 1)
    sigma = 1.7
    k1 = S.gaussian_kernel_1d(sigma)
    x = np.arange(-(len(k1) // 2), len(k1) // 2 + 1)
    assert abs(k1.sum() - 1) < 1e-12
    assert abs(np.sqrt((k1 * x * x).sum()) - sigma) < 0.08
    assert np.allclose(S.smooth_gaussian(g, sigma), ndi.gaussian_filter(g, sigma, mode="reflect", truncate=(len(k1) // 2) / sigma), atol=1e-9)


def test_smoothing_keeps_constant_image_constant_and_size():
    g = np.full((20, 25), 90.0)
    for kern in ("standard", "original"):
        assert np.allclose(S.smooth_gaussian(g, 1.0, kern), 90.0)


def test_sobel_matches_loop_in_interior():
    g = np.rint(blocky(seed=2))
    sx, sy = S.sobel(g)
    assert np.array_equal(sx[1:-1, 1:-1], ref.ref_correlate_valid(S.SOBEL_X, g, 1))
    assert np.array_equal(sy[1:-1, 1:-1], ref.ref_correlate_valid(S.SOBEL_Y, g, 1))


def test_magnitude_direction_matches_loop_and_has_no_warnings():
    g = np.rint(blocky(seed=3))
    sx, sy = S.sobel(g)
    with np.errstate(all="raise"):
        mag, deg = S.magnitude_direction(sx, sy)
    rm, rd = ref.ref_magnitude_direction(sx, sy)
    assert np.allclose(mag, rm) and np.array_equal(deg, rd)


@pytest.mark.parametrize("ties", ["keep_one", "suppress_all"])
def test_nms_matches_loop(ties):
    g = np.rint(blocky(seed=4))
    mag, deg = S.magnitude_direction(*S.sobel(g))
    assert np.array_equal(S.non_max_suppression(mag, deg, ties), ref.ref_nms(mag, deg, ties))


def test_keep_one_keeps_a_symmetric_step_edge_but_original_rule_removes_it():
    g = np.zeros((12, 12)); g[:, 6:] = 100.0
    mag, deg = S.magnitude_direction(*S.sobel(g))
    assert S.non_max_suppression(mag, deg, "keep_one")[5].astype(bool).sum() == 1
    assert S.non_max_suppression(mag, deg, "suppress_all")[5].astype(bool).sum() == 0


def test_double_threshold_matches_loop_even_for_odd_thresholds():
    img = np.random.default_rng(5).uniform(0, 400, (20, 25))
    for lo, hi in [(20, 40), (100, 300), (50, 255)]:
        assert np.array_equal(S.double_threshold(img, lo, hi), ref.ref_threshold(img, lo, hi))


def test_hysteresis_modes_match_loops():
    thr = np.random.default_rng(6).choice([0, 75, 255], (25, 30), p=[.6, .25, .15]).astype(np.uint8)
    thr[0, :] = thr[-1, :] = thr[:, 0] = thr[:, -1] = 0
    single, full = S.hysteresis(thr, "single_pass"), S.hysteresis(thr, "full")
    assert np.array_equal(single, ref.ref_hysteresis_single_pass(thr))
    assert np.array_equal(full, ref.ref_hysteresis_full(thr))
    assert (full >= single).all()


def test_pipeline_matches_loop_reference_on_real_crop():
    g = to_gray(IMG("Lenna.png"))[100:164, 100:164]
    res = canny_improved(g, kernel="original", quantize="trunc", nms_ties="suppress_all")
    sm = ref.ref_quantize(ref.ref_correlate_valid(S.gaussian_kernel(1.0), g, S.K), "trunc")
    sx, sy = ref.ref_correlate_valid(S.SOBEL_X, sm, 1), ref.ref_correlate_valid(S.SOBEL_Y, sm, 1)
    mag, deg = ref.ref_magnitude_direction(sx, sy)
    want = ref.ref_hysteresis_single_pass(ref.ref_threshold(ref.ref_nms(mag, deg, "suppress_all"), 20, 40))
    got = res.edges[4:-4, 4:-4]
    # compare away from the reflected border band; the loop reference only exists there
    assert np.array_equal(got[4:-4, 4:-4], want[4:-4, 4:-4])


def test_improved_equals_original_script_except_for_known_fixes():
    """Replaying the original in-place suppression on the corrected earlier stages
    reproduces the original script's interior output exactly."""
    import canny_original as co

    gray = to_gray(IMG("bird 2.jpg"))
    import PIL.Image
    small = np.asarray(PIL.Image.fromarray(gray.astype(np.uint8)).resize((120, 80)), dtype=np.float64)
    orig = B.original_manual_canny(small, 1, 20, 40)
    sm = S.quantize(S.smooth_gaussian(small, 1.0, "original"), "trunc")
    mag, deg = S.magnitude_direction(*S.sobel(sm))
    nms = co.nonMaximalSupress(mag.copy(), deg)
    replay = S.hysteresis(S.double_threshold(nms, 20, 40))
    m = 20
    assert np.array_equal((orig[m:-m, m:-m] > 0), (replay[m:-m, m:-m] > 0))


# ---------- denoising ----------

@pytest.mark.parametrize("method", sorted(D.METHODS))
def test_denoisers_keep_shape_and_range(method):
    g = blocky()
    out = D.denoise(g, method)
    assert out.shape == g.shape and np.isfinite(out).all()
    assert out.min() >= -1 and out.max() <= 256


def test_median_matches_scipy_and_removes_salt_and_pepper():
    g = np.full((20, 20), 100.0)
    g[5, 5], g[10, 10] = 255, 0
    out = D.median(g, 3)
    assert out[5, 5] == 100 and out[10, 10] == 100


def test_bilateral_is_close_to_opencv():
    import cv2

    g = np.rint(blocky(40, 50, 7)).astype(np.uint8)
    mine = D.bilateral(g.astype(np.float64), radius=2, sigma_space=2.0, sigma_color=25.0)
    theirs = cv2.bilateralFilter(g, 5, 25.0, 2.0).astype(np.float64)
    assert np.abs(mine - theirs).mean() < 1.5


def test_denoisers_reduce_error_on_noisy_scene():
    clean, _ = synthetic_scene(128, 3)
    noisy = add_noise(clean, 20, 1)
    base = ((noisy - clean) ** 2).mean()
    for m in ("gaussian", "median", "bilateral", "nlm"):
        assert ((D.denoise(noisy, m) - clean) ** 2).mean() < base, m


def test_unknown_denoiser_raises():
    with pytest.raises(ValueError):
        D.denoise(blocky(), "wavelet")


# ---------- thresholds ----------

def test_threshold_modes():
    sm = np.full((10, 10), 100.0)
    assert select_thresholds("fixed", sm, sm, 5, 9) == (5.0, 9.0)
    lo, hi = select_thresholds("median", sm, sm)
    assert lo == pytest.approx(67.0) and hi == pytest.approx(133.0)
    nms = np.zeros((10, 10)); nms[0, :5] = 10; nms[1, :5] = 200
    lo, hi = select_thresholds("otsu", sm, nms)
    assert 10 < hi < 200 and lo == pytest.approx(hi / 2)
    assert select_thresholds("otsu", sm, np.zeros((5, 5)), 7, 9) == (7.0, 9.0)
    with pytest.raises(ValueError):
        select_thresholds("magic", sm, sm)


# ---------- pipeline, baselines, metrics ----------

@pytest.mark.parametrize("den", ["gaussian", "median", "bilateral", "nlm", "none"])
@pytest.mark.parametrize("thr", ["fixed", "median", "otsu"])
def test_every_option_combination_runs(den, thr):
    img, _ = synthetic_scene(96, 0)
    r = canny_improved(img, denoise=den, threshold=thr, keep_stages=True)
    assert r.edges.shape == img.shape and set(np.unique(r.edges)) <= {0, 255}
    assert r.thresholds[0] <= r.thresholds[1]
    assert not r.edges[0].any() and not r.edges[-1].any() and not r.edges[:, 0].any() and not r.edges[:, -1].any()
    assert {"gray", "smoothed", "magnitude", "nms", "thresholded"} <= set(r.stages)
    assert r.timings["total"] > 0


def test_pipeline_accepts_paths_rgb_and_gray_and_is_deterministic():
    a = canny_improved(IMG("Lenna.png")).edges
    b = canny_improved(IMG("Lenna.png")).edges
    assert np.array_equal(a, b) and a.shape == (512, 512)
    assert canny_improved(IMG("cameraman.jpg")).edges.shape == (512, 512)


def test_pipeline_rejects_tiny_images_and_bad_options():
    with pytest.raises(ValueError):
        canny_improved(np.zeros((4, 4)))
    with pytest.raises(ValueError):
        canny_improved(blocky(), hysteresis="nope")
    with pytest.raises(ValueError):
        canny_improved(blocky(), quantize="nope")


def test_baselines_return_binary_maps_of_same_size():
    img, _ = synthetic_scene(96, 1)
    g = img.astype(np.float64)
    for fn in (B.opencv_canny, B.sobel_edges, B.prewitt_edges, B.laplacian_edges):
        out = fn(g)
        assert out.shape == g.shape and out.dtype == np.uint8 and set(np.unique(out)) <= {0, 255} and out.any(), fn.__name__
    assert B.sobel_edges(np.full((20, 20), 50.0)).sum() == 0


def test_metrics_known_cases():
    gt = np.zeros((20, 20), np.uint8); gt[10, 2:18] = 255
    assert edge_prf(gt, gt)["f1"] == 1.0
    shifted = np.roll(gt, 1, axis=0)
    assert edge_prf(shifted, gt, tol=0)["f1"] == 0.0 and edge_prf(shifted, gt, tol=1)["f1"] == 1.0
    half = gt.copy(); half[10, 10:] = 0
    r = edge_prf(half, gt, tol=0)
    assert r["precision"] == 1.0 and r["recall"] == pytest.approx(0.5)
    z = np.zeros_like(gt)
    assert edge_prf(z, gt)["f1"] == 0.0 and edge_prf(z, z)["f1"] == 1.0
    assert edge_density(gt) == pytest.approx(16 / 400)


def test_synthetic_scene_and_degradations():
    img, gt = synthetic_scene(128, 4)
    assert img.shape == gt.shape == (128, 128) and gt.any()
    assert np.array_equal(img, synthetic_scene(128, 4)[0])
    assert add_noise(img, 0, 0).std() == pytest.approx(img.std())
    assert change_contrast(img, 0.5).std() < img.std()
    assert change_brightness(img, 0.5).mean() > img.mean()
    assert resize(img, 64).shape == (64, 64)
    assert edge_prf(canny_improved(img).edges, gt)["f1"] > 0.8      # clean drawn scene


def test_streamlit_app_starts():
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(os.path.join(ROOT, "app.py"), default_timeout=60).run()
    assert not at.exception

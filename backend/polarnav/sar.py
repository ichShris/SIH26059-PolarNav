"""SAR iceberg detection: two-parameter CA-CFAR on Sentinel-1-like intensity scenes.

CFAR is the classical, training-free baseline for iceberg/ship detection in SAR and
the stage that proposes candidate chips for a learned detector (YOLOv8 in the full
design) once labelled Sentinel-1 chips are available. The scene generator produces
multilook gamma speckle over wind-modulated open water, textured pack ice with floes,
and icebergs of varying size and brightness, with known ground truth for scoring.
"""
from __future__ import annotations

import base64

import numpy as np
from scipy import ndimage
from scipy.special import gammaincinv

LOOKS = 4
PIXEL_M = 40.0


def _db(x):
    return 10 ** (np.asarray(x) / 10.0)


def scene(seed: int, n: int = 256) -> dict:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:n, 0:n]
    wind_mod = ndimage.gaussian_filter(rng.standard_normal((n, n)), 18) * 25
    sigma0 = _db(-19.0 + wind_mod)
    # pack-ice region with a ragged edge, floes and leads
    edge = n * rng.uniform(0.35, 0.65) + ndimage.gaussian_filter1d(rng.standard_normal(n), 6) * 60
    ice = xx > edge[yy]
    floes = ndimage.gaussian_filter(rng.standard_normal((n, n)), 3) > 0.02
    sigma0 = np.where(ice & floes, _db(-13.5 + rng.normal(0, 1.2, (n, n))), sigma0)
    sigma0 = np.where(ice & ~floes, _db(-21.0), sigma0)
    truth = []
    for _ in range(rng.integers(7, 13)):
        L = rng.uniform(3, 22)
        W = L * rng.uniform(0.4, 0.9)
        cx, cy = rng.uniform(12, n - 12, 2)
        ang = rng.uniform(0, np.pi)
        u = (xx - cx) * np.cos(ang) + (yy - cy) * np.sin(ang)
        v = -(xx - cx) * np.sin(ang) + (yy - cy) * np.cos(ang)
        blob = (u / (L / 2)) ** 2 + (v / (W / 2)) ** 2 <= 1.0
        if blob.sum() < 3:
            continue
        sigma0 = np.where(blob, _db(rng.uniform(-4.0, 3.0)), sigma0)
        ys, xs = np.nonzero(blob)
        truth.append({"x0": int(xs.min()), "y0": int(ys.min()), "x1": int(xs.max()), "y1": int(ys.max()),
                      "length_m": float(L * PIXEL_M), "in_pack": bool(ice[int(cy), int(cx)])})
    intensity = sigma0 * rng.gamma(LOOKS, 1.0 / LOOKS, (n, n))
    return {"intensity": intensity.astype(np.float32), "truth": truth}


def ca_cfar(intensity: np.ndarray, guard: int = 4, train: int = 10, pfa: float = 1e-5) -> np.ndarray:
    """Cell-averaging CFAR for L-look gamma clutter."""
    outer = 2 * (guard + train) + 1
    inner = 2 * guard + 1
    s_out = ndimage.uniform_filter(intensity, outer, mode="reflect") * outer ** 2
    s_in = ndimage.uniform_filter(intensity, inner, mode="reflect") * inner ** 2
    clutter = (s_out - s_in) / (outer ** 2 - inner ** 2)
    alpha = gammaincinv(LOOKS, 1.0 - pfa) / LOOKS
    return intensity > alpha * clutter


def detect(intensity: np.ndarray, pfa: float = 1e-5, min_px: int = 4, min_mean_db: float = -9.0) -> tuple[list[dict], int]:
    """Stage 1: CA-CFAR prescreen. Stage 2: discriminate on size and mean backscatter.

    Returns (accepted detections, number of CFAR candidates before discrimination).
    """
    mask = ca_cfar(intensity, pfa=pfa)
    mask = ndimage.binary_closing(mask, iterations=1)
    lab, _ = ndimage.label(mask)
    out, candidates = [], 0
    for k, sl in enumerate(ndimage.find_objects(lab), start=1):
        blob = lab[sl] == k
        px = int(blob.sum())
        candidates += 1
        mean_db = 10 * np.log10(float(intensity[sl][blob].mean()) + 1e-12)
        if px < min_px or mean_db < min_mean_db:
            continue
        ys, xs = sl
        out.append({"x0": xs.start, "y0": ys.start, "x1": xs.stop - 1, "y1": ys.stop - 1,
                    "area_m2": px * PIXEL_M ** 2,
                    "length_m": float(max(xs.stop - xs.start, ys.stop - ys.start) * PIXEL_M),
                    "mean_db": round(mean_db, 1)})
    return out, candidates


def _overlap(a, b) -> bool:
    return not (a["x1"] < b["x0"] or b["x1"] < a["x0"] or a["y1"] < b["y0"] or b["y1"] < a["y0"])


def run(seed: int, pfa: float = 1e-5) -> dict:
    sc = scene(seed)
    det, candidates = detect(sc["intensity"], pfa=pfa)
    tp_truth = sum(1 for t in sc["truth"] if any(_overlap(t, d) for d in det))
    tp_det = sum(1 for d in det if any(_overlap(t, d) for t in sc["truth"]))
    for d in det:
        d["match"] = any(_overlap(t, d) for t in sc["truth"])
    db = 10 * np.log10(sc["intensity"] + 1e-6)
    img = np.clip((db + 28.0) / 32.0 * 255.0, 0, 255).astype(np.uint8)
    n = img.shape[0]
    return {
        "seed": seed, "size": n, "pixel_m": PIXEL_M, "pfa": pfa,
        "image_b64": base64.b64encode(img.tobytes()).decode(),
        "detections": det, "truth": sc["truth"],
        "scores": {"recall": tp_truth / max(len(sc["truth"]), 1),
                   "precision": tp_det / max(len(det), 1),
                   "n_truth": len(sc["truth"]), "n_detections": len(det), "n_cfar_candidates": candidates},
    }


def benchmark(n_scenes: int = 40, pfa: float = 1e-5) -> dict:
    r, p = [], []
    for s in range(n_scenes):
        out = run(1000 + s, pfa)
        r.append(out["scores"]["recall"])
        p.append(out["scores"]["precision"])
    return {"scenes": n_scenes, "pfa": pfa, "recall": float(np.mean(r)), "precision": float(np.mean(p))}

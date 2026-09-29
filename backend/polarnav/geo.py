"""Coastline handling: Natural Earth polygons -> projected rings, land mask, coast distance."""
from __future__ import annotations

import json
from functools import lru_cache

import numpy as np
from scipy import ndimage

from . import config as C
from .projection import to_xy

LAND_FILE = C.DATA_DIR / "ne_50m_land_south.geojson"
SHELF_FILE = C.DATA_DIR / "ne_50m_antarctic_ice_shelves_south.geojson"


def _rings(path) -> list[np.ndarray]:
    """All outer/inner rings of a GeoJSON file as (N, 2) arrays of (lon, lat)."""
    with open(path, encoding="utf-8") as f:
        gj = json.load(f)
    rings = []
    for feat in gj["features"]:
        g = feat["geometry"]
        polys = [g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]
        for poly in polys:
            for ring in poly:
                rings.append(np.asarray(ring, dtype=float)[:, :2])
    return rings


@lru_cache(maxsize=None)
def projected_rings(kind: str = "land") -> tuple[np.ndarray, ...]:
    path = LAND_FILE if kind == "land" else SHELF_FILE
    out = []
    for r in _rings(path):
        # Anything north of ~15 S projects far outside the domain; clamp so the
        # ring stays closed but cheap.
        lat = np.clip(r[:, 1], -90.0, -10.0)
        x, y = to_xy(lat, r[:, 0])
        out.append(np.column_stack([x, y]))
    return tuple(out)


def rasterize(rings, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Even-odd scanline fill of polygon rings onto cell centres. Returns bool[ny, nx]."""
    mask = np.zeros((len(ys), len(xs)), dtype=bool)
    x0 = np.concatenate([r[:-1, 0] for r in rings])
    y0 = np.concatenate([r[:-1, 1] for r in rings])
    x1 = np.concatenate([r[1:, 0] for r in rings])
    y1 = np.concatenate([r[1:, 1] for r in rings])
    for j, yc in enumerate(ys):
        hit = (y0 <= yc) != (y1 <= yc)
        if not hit.any():
            continue
        t = (yc - y0[hit]) / (y1[hit] - y0[hit])
        xc = np.sort(x0[hit] + t * (x1[hit] - x0[hit]))
        # count of crossings left of each cell centre -> odd means inside
        cnt = np.searchsorted(xc, xs)
        mask[j] = (cnt % 2) == 1
    return mask


def build_masks() -> dict[str, np.ndarray]:
    land = rasterize(projected_rings("land"), C.NAV_X, C.NAV_Y)
    shelf = rasterize(projected_rings("shelf"), C.NAV_X, C.NAV_Y)
    blocked = land | shelf
    # Antarctica (continent + attached shelves) is the connected blob under the pole.
    labels, _ = ndimage.label(blocked)
    antarctica = labels == labels[C.NAV_N // 2, C.NAV_N // 2]
    # distance (km) from every cell to the Antarctic coast / ice-shelf front
    coast_km = ndimage.distance_transform_edt(~antarctica) * C.CELL_KM
    return {"land": land, "shelf": shelf, "blocked": blocked, "antarctica": antarctica,
            "coast_km": coast_km.astype(np.float32)}


def _simplify(pts: np.ndarray, tol: float) -> np.ndarray:
    """Radial-distance thinning; good enough for display polylines."""
    keep = [0]
    last = pts[0]
    for i in range(1, len(pts) - 1):
        if np.hypot(*(pts[i] - last)) >= tol:
            keep.append(i)
            last = pts[i]
    keep.append(len(pts) - 1)
    return pts[keep]


def display_polylines(tol_km: float = 12.0) -> dict[str, list]:
    lim = C.NAV_HALF_KM * 1.05
    out = {}
    for kind in ("land", "shelf"):
        lines = []
        for r in projected_rings(kind):
            if (np.abs(r) > lim).all(axis=1).all():
                continue
            if r[:, 0].max() < -lim or r[:, 0].min() > lim or r[:, 1].max() < -lim or r[:, 1].min() > lim:
                continue
            s = _simplify(r, tol_km)
            if len(s) >= 3:
                lines.append(np.round(s, 1).tolist())
        out[kind] = lines
    return out

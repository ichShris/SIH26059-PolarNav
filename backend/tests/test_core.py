"""Unit tests for the physics, risk, routing and packing building blocks.

These avoid the 4-year dataset so they run in seconds:  python -m pytest backend/tests
"""
import sys
import zlib
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from polarnav import config as C  # noqa: E402
from polarnav import edge_sync, fuel, polaris, sar  # noqa: E402
from polarnav.projection import bearing_deg, scale_factor, to_latlon, to_xy  # noqa: E402


# ------------------------------------------------------------------ projection
def test_projection_roundtrip():
    lat = np.array([-89.0, -71.0, -60.0, -34.0])
    lon = np.array([-170.0, 0.0, 76.19, 18.4])
    x, y = to_xy(lat, lon)
    la, lo = to_latlon(x, y)
    np.testing.assert_allclose(la, lat, atol=1e-9)
    np.testing.assert_allclose(lo, lon, atol=1e-9)


def test_true_scale_at_71s_and_orientation():
    assert scale_factor(-71.0) == pytest.approx(1.0, abs=1e-12)
    # EPSG:3031 orientation: 0 deg at +y, 90E at +x
    x, y = to_xy(-70.0, 90.0)
    assert x > 0 and abs(y) < 1e-6
    # due north from a point is bearing 0; east is 90
    x0, y0 = to_xy(-65.0, 30.0)
    xn, yn = to_xy(-64.0, 30.0)
    xe, ye = to_xy(-65.0, 30.5)
    assert bearing_deg(x0, y0, xn, yn) == pytest.approx(0.0, abs=0.5)
    assert bearing_deg(x0, y0, xe, ye) == pytest.approx(90.0, abs=0.5)


# ------------------------------------------------------------------ POLARIS
def test_polaris_open_water_is_normal():
    r = polaris.rio(np.array([0.0]), np.array([0.0]), "PC5")
    assert r[0] == 30  # 10 tenths of ice-free water x RIV 3
    assert polaris.operation_level(r, "PC5")[0] == polaris.NORMAL


def test_polaris_levels_follow_ice_class():
    sic = np.array([1.0])
    thick = np.array([2.8])  # light multi-year ice
    r_pc5 = polaris.rio(sic, thick, "PC5")
    r_pc1 = polaris.rio(sic, thick, "PC1")
    r_nic = polaris.rio(sic, thick, "No Ice Class")
    assert r_pc1 > r_pc5 > r_nic
    assert polaris.operation_level(r_pc1, "PC1")[0] == polaris.NORMAL
    assert polaris.operation_level(r_nic, "No Ice Class")[0] == polaris.SPECIAL
    # a non-Polar-Class ship never gets "elevated": negative RIO is special consideration
    assert polaris.operation_level(np.array([-3.0]), "IA")[0] == polaris.SPECIAL
    assert polaris.operation_level(np.array([-3.0]), "PC7")[0] == polaris.ELEVATED


def test_polaris_ice_type_bins():
    t = polaris.ice_type(np.array([0.05, 0.12, 0.2, 0.6, 1.1, 1.5, 3.5]))
    names = [polaris.ICE_TYPES[i] for i in t]
    assert names == ["New ice", "Grey ice", "Grey-white ice", "Thin FY 2nd stage", "Medium FY",
                     "Thick FY", "Heavy multi-year"]


# ------------------------------------------------------------------ fuel
def test_attainable_speed_calibration():
    # rated capability: ~3 kn in level ice of the rated thickness at full power
    for cls in ("PC5", "PC3"):
        cap = C.ICE_CLASSES[cls].capability_m
        assert fuel.attainable_speed_kn(np.array([cap]), cap)[0] == pytest.approx(3.0, abs=0.05)
    assert fuel.attainable_speed_kn(np.array([0.0]), 1.0)[0] == pytest.approx(C.VESSEL.max_speed_kn, abs=0.05)


def test_fuel_per_km_higher_in_ice():
    v, fpk, hpk, beset = fuel.cell_performance(np.array([0.0, 0.9]), np.array([0.0, 0.8]), "PC5", 12.0,
                                               np.array([False, False]))
    assert v[0] == pytest.approx(12.0)
    assert v[1] < v[0]
    assert fpk[1] > fpk[0]
    assert not beset.any()


# ------------------------------------------------------------------ SAR
def test_cfar_finds_bright_targets_in_clutter():
    rng = np.random.default_rng(0)
    img = 0.01 * rng.gamma(4, 0.25, (128, 128)).astype(np.float32)
    img[60:64, 40:44] = 1.0
    img[20:23, 100:104] = 0.8
    det, cand = sar.detect(img, pfa=1e-6)
    assert len(det) == 2 and cand >= 2


def test_sar_benchmark_quality():
    b = sar.benchmark(8)
    assert b["recall"] > 0.9 and b["precision"] > 0.9


# ------------------------------------------------------------------ edge sync
def test_bundle_packs_and_decompresses():
    n = C.ICE_N
    rng = np.random.default_rng(1)
    sics = [np.clip(rng.random((n, n)), 0, 1).astype(np.float32) * (rng.random((n, n)) > 0.7) for _ in range(4)]
    rios = [polaris.rio(s, np.full_like(s, 0.8), "PC5") for s in sics]
    bergs = [{"id": "IB-101", "forecast": [[10.0 * k, 5.0 * k] for k in range(25)],
              "ellipses": [{"a": 1.0 + k} for k in range(13)]}]
    wps = [{"name": "P1", "lat": -60.0, "lon": 10.0, "t_h": 0.0}]
    packed, stats = edge_sync.pack("2025-11-20", "PC5", sics, rios, bergs, wps)
    raw = zlib.decompress(packed)
    hlen = int.from_bytes(raw[:4], "little")
    assert b'"ice_class":"PC5"' in raw[4:4 + hlen]
    assert stats["packed_bytes"] < stats["raw_bytes"]


# ------------------------------------------------------------------ geography
def test_land_mask_and_places():
    from polarnav import geo
    m = geo.build_masks()
    c = C.NAV_N // 2
    assert m["blocked"][c, c]  # South Pole is on the continent
    # the approach waypoint off Maitri is ocean
    x, y = to_xy(-58.0, 14.0)
    i = int((x + C.NAV_HALF_KM) // C.CELL_KM)
    j = int((y + C.NAV_HALF_KM) // C.CELL_KM)
    assert not m["blocked"][j, i]
    frac = m["antarctica"].sum() * C.CELL_KM ** 2 / 1e6
    assert 12 < frac < 16.5  # Antarctica incl. ice shelves ~14 M km^2 (projected area)


# ------------------------------------------------------------------ live data plumbing (no network)
def test_nsidc_projection_matches_published_grid_corners():
    from polarnav.live import nsidc_xy
    # NSIDC 25 km south grid outer corners (EPSG:3412)
    for lat, lon, (ex, ey) in ((-39.23, 317.76, (-3950, 4350)), (-41.45, 135.0, (3950, -3950))):
        x, y = nsidc_xy(lat, lon)
        assert abs(x - ex) < 1.0 and abs(y - ey) < 1.0


def test_regrid_reads_concentration_and_ignores_flags():
    from polarnav.live import _Regrid
    raw = np.full((332, 316), 1000, dtype=np.uint16)  # 100 % everywhere ...
    raw[:, :158] = 2540                                # ... except flagged land on the western half
    out = _Regrid()(raw)
    assert out.shape == (C.ICE_N, C.ICE_N)
    assert out.max() == pytest.approx(1.0)
    assert out[:, : C.ICE_N // 2 - 2].max() == 0.0     # flags never become ice

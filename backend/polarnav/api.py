"""PolarNav HTTP API (FastAPI). Also serves the built dashboard from frontend/dist.

Every data endpoint takes ``mode``: ``sim`` (the synthetic twin, any date 2022-2025, with a
verifying truth) or ``live`` (latest NSIDC sea ice, ECMWF winds, Open-Meteo currents and
USNIC icebergs; no future truth exists yet).
"""
from __future__ import annotations

import base64
import datetime as dt
import json
import threading
import time
from functools import lru_cache

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import config as C
from . import edge_sync, geo, polaris, routing, sar
from .data import get_store
from .drift import METRICS_FILE as DRIFT_METRICS, DriftEngine
from .live import BERG_HEIGHT_M, get_live
from .sic_model import Forecaster, skill

app = FastAPI(title="PolarNav API", version="0.2.0",
              description="AI-enabled Antarctic sea-ice, iceberg trajectory and navigation decision support")
app.add_middleware(GZipMiddleware, minimum_size=2000)

_lock = threading.Lock()
_svc: dict = {}


def svc() -> dict:
    with _lock:
        if not _svc:
            store = get_store()
            _svc["store"] = store
            _svc["fc"] = Forecaster(store)
            _svc["drift"] = DriftEngine(store)
        return _svc


class Ctx:
    """One data world: the synthetic twin ('sim') or a live snapshot ('live:<fetch time>')."""

    def __init__(self, key: str, store, fc, drift, live: bool):
        self.key, self.store, self.fc, self.drift, self.live = key, store, fc, drift, live


_ctx: dict[str, Ctx] = {}
_ctx_lock = threading.Lock()


def ctx(mode: str = "sim", refresh: bool = False) -> Ctx:
    if mode not in ("sim", "live"):
        raise HTTPException(400, "mode must be 'sim' or 'live'")
    s = svc()
    if mode == "sim":
        return _ctx.setdefault("sim", Ctx("sim", s["store"], s["fc"], s["drift"], False))
    try:
        ls = get_live(s["store"], refresh=refresh)
    except Exception as e:
        raise HTTPException(503, f"live data sources unreachable: {e}")
    key = f"live:{ls.fetched_at}"
    with _ctx_lock:
        if key not in _ctx:
            for k in [k for k in _ctx if k.startswith("live:")]:
                del _ctx[k]  # drop the previous live snapshot
            _ctx[key] = Ctx(key, ls, Forecaster(ls), DriftEngine(ls), True)
        return _ctx[key]


def b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def q_sic(a: np.ndarray) -> str:
    return b64(np.round(np.clip(a, 0, 1) * 100).astype(np.uint8))


def parse_day(date: str | None, c: Ctx) -> int:
    if c.live:
        return c.store.day0  # live mode always runs from the latest satellite analysis
    d = C.DEFAULT_DATE if not date else dt.date.fromisoformat(date)
    day = C.day_index(d)
    lo, hi = C.HISTORY_DAYS - 1, c.store.n_days - 1 - C.FORECAST_DAYS
    if not lo <= day <= hi:
        raise HTTPException(400, f"date must be between {C.index_day(lo)} and {C.index_day(hi)}")
    return day


# ------------------------------------------------------------------ cached computations (keyed by data world)
@lru_cache(maxsize=1)
def sar_benchmark():
    return sar.benchmark(20)


@lru_cache(maxsize=24)
def sic_bundle(key: str, day: int):
    c = _ctx[key]
    return [c.store.sic_obs(day)] + list(c.fc.forecast(day))


@lru_cache(maxsize=12)
def berg_scenario(key: str, day: int) -> dict:
    c = _ctx[key]
    if c.live:
        return c.drift.live_scenario(day, c.store.icebergs["bergs"], sic_bundle(key, day), BERG_HEIGHT_M)
    return c.drift.scenario(day, np.stack(sic_bundle(key, day)[1:]))


@lru_cache(maxsize=48)
def cached_plan(key, day, o_lat, o_lon, d_lat, d_lon, ice_class, cruise_kn, w_time, w_risk, avoid_bergs, w_ice,
                berg_margin_nm):
    req = routing.RouteRequest(day=day, origin=(o_lat, o_lon), dest=(d_lat, d_lon), ice_class=ice_class,
                               cruise_kn=cruise_kn, w_time=w_time, w_risk=w_risk, avoid_bergs=avoid_bergs,
                               w_ice=w_ice, berg_margin_nm=berg_margin_nm)
    t0 = time.time()
    out = routing.plan(_ctx[key].store, sic_bundle(key, day), req, berg_scenario(key, day))
    out["compute_s"] = round(time.time() - t0, 2)
    return out


def live_check(c: Ctx, day: int) -> dict:
    """Verification that is possible in real time: earlier forecasts valid *today* vs today's analysis.

    The forecast issued k days ago for today (+24k h) is compared with today's satellite
    analysis. Unlike the hindcast error layer (forecast vs future truth), this needs no
    information from the future, so it runs every day on the ship or ashore.
    """
    analysis = sic_bundle(c.key, day)[0]
    ocean = ~c.store.ice_blocked
    rows, err24 = [], None
    for k in range(1, C.FORECAST_DAYS + 1):
        issue = day - k
        if issue < C.HISTORY_DAYS - 1:
            continue
        fc = sic_bundle(c.key, issue)[k]
        pers = sic_bundle(c.key, issue)[0]
        rows.append({"lead_h": 24 * k, "issued": str(C.index_day(issue)),
                     "model": skill(fc, analysis, ocean), "persistence": skill(pers, analysis, ocean)})
        if k == 1:
            err24 = np.clip(np.round((fc - analysis) * 100), -127, 127).astype(np.int8)
    note = ("Past issue days use the analysed winds (archived forecasts are not fetched), which slightly "
            "flatters the model." if c.live else None)
    return {"per_lead": rows, "error_24h": b64(err24) if err24 is not None else None,
            "reference": "today's satellite analysis", "note": note}


def rio_slices(c: Ctx, day: int, ice_class: str):
    return [polaris.rio(f, c.store.thickness(f, day + k), ice_class) for k, f in enumerate(sic_bundle(c.key, day))]


def sync_bundle(c: Ctx, day: int, ice_class: str, plan: dict):
    wps = plan["optimal"]["waypoints"] if plan.get("optimal") else []
    return edge_sync.pack(str(C.index_day(day)), ice_class, sic_bundle(c.key, day), rio_slices(c, day, ice_class),
                          berg_scenario(c.key, day)["bergs"], wps)


# ------------------------------------------------------------------ endpoints
@app.get("/api/health")
def health():
    return {"ok": True, "version": app.version}


@app.get("/api/meta")
def meta():
    s = svc()
    store = s["store"]
    drift_m = json.loads(DRIFT_METRICS.read_text()) if DRIFT_METRICS.exists() else None
    bench_f = C.ARTIFACT_DIR / "route_benchmark.json"
    bench = json.loads(bench_f.read_text())["summary"] if bench_f.exists() else None
    return {
        "grid": {"cell_km": C.CELL_KM, "nav_n": C.NAV_N, "nav_half_km": C.NAV_HALF_KM,
                 "ice_n": C.ICE_N, "ice_off": C.ICE_OFF, "row0": "south (min y)",
                 "projection": "South polar stereographic, true scale 71S (EPSG:3031, spherical)"},
        "dates": {"min": str(C.index_day(C.HISTORY_DAYS - 1)),
                  "max": str(C.index_day(store.n_days - 1 - C.FORECAST_DAYS)),
                  "default": str(C.DEFAULT_DATE), "test_from": str(C.VAL_END + dt.timedelta(days=1))},
        "ice_classes": [{"name": c.name, "polar_class": c.polar_class, "capability_m": c.capability_m,
                         "elevated_speed_kn": c.elevated_speed_kn} for c in C.ICE_CLASSES.values()],
        "places": C.PLACES,
        "vessel": C.VESSEL.__dict__,
        "polaris": {"ice_types": polaris.ICE_TYPES, "riv": polaris.RIV},
        "metrics": {"sic": s["fc"].metrics, "drift": drift_m, "routes": bench,
                    "sar": sar_benchmark()},
        "data_mode": "synthetic-digital-twin",
        "modes": ["sim", "live"],
    }


@app.get("/api/coastline")
def coastline():
    return geo.display_polylines()


@app.get("/api/coastline/globe")
def coastline_globe():
    """All continents (Natural Earth 1:50m, thinned) as lon/lat polygons for the 3-D globe."""
    return FileResponse(C.DATA_DIR / "globe_land.json", media_type="application/json")


@app.get("/api/scenario")
def scenario(date: str | None = None, mode: str = "sim"):
    c = ctx(mode)
    day = parse_day(date, c)
    store = c.store
    sics = sic_bundle(c.key, day)
    thick = [store.thickness(f, day + k) for k, f in enumerate(sics)]
    # a verifying truth for the forecast days only exists in hindcast (simulation) mode
    truth = [] if c.live else [store.sic_truth(day + 1 + k) for k in range(C.FORECAST_DAYS)]
    w = store.wind_truth(day)[:, ::2, ::2]  # 200 km arrows
    bergs = berg_scenario(c.key, day)
    area = lambda f: float((f > 0.15).sum() * C.CELL_KM ** 2)
    return {
        "mode": "live" if c.live else "sim",
        "date": str(C.index_day(day)),
        "day": day,
        "is_test_period": (not c.live) and C.index_day(day) > C.VAL_END,
        "sic": [q_sic(f) for f in sics],
        "sic_truth": [q_sic(f) for f in truth],
        "thickness": [b64(np.round(np.clip(h, 0, 5.1) * 50).astype(np.uint8)) for h in thick],
        "extent_km2": [area(f) for f in sics],
        "wind": {"n": int(w.shape[1]), "step_km": C.CELL_KM * 4, "u": np.round(w[0], 1).ravel().tolist(),
                 "v": np.round(w[1], 1).ravel().tolist()},
        "bergs": bergs,
        "skill": [] if c.live else c.fc.daily_skill(day, np.stack(sics[1:])),
        "live_check": live_check(c, day),
        "sources": store.sources() if c.live else None,
    }


@app.get("/api/live/status")
def live_status(refresh: bool = False):
    """Fetch (or reuse, at most 1 h old) the live feeds and report what they contain."""
    return ctx("live", refresh=refresh).store.sources()


@app.get("/api/polaris")
def polaris_grid(date: str | None = None, ice_class: str = "PC5", mode: str = "sim"):
    if ice_class not in C.ICE_CLASSES:
        raise HTTPException(400, "unknown ice class")
    c = ctx(mode)
    day = parse_day(date, c)
    rs = rio_slices(c, day, ice_class)
    return {"ice_class": ice_class, "rio": [b64(np.clip(np.round(r), -127, 127).astype(np.int8)) for r in rs],
            "level": [b64(polaris.operation_level(r, ice_class).astype(np.uint8)) for r in rs]}


class LatLon(BaseModel):
    lat: float = Field(ge=-80, le=-10)
    lon: float = Field(ge=-180, le=360)


class RouteBody(BaseModel):
    date: str | None = None
    origin: LatLon
    dest: LatLon
    ice_class: str = "PC5"
    cruise_kn: float = Field(12.0, ge=4, le=16)
    w_time: float = Field(0.4, ge=0, le=5)
    w_risk: float = Field(0.02, ge=0, le=1)
    avoid_bergs: bool = True
    w_ice: float = Field(0.0, ge=0, le=2)
    berg_margin_nm: float = Field(5.0, ge=1, le=30)
    mode: str = "sim"


def _plan(body: RouteBody):
    if body.ice_class not in C.ICE_CLASSES:
        raise HTTPException(400, "unknown ice class")
    c = ctx(body.mode)
    day = parse_day(body.date, c)
    return c, day, cached_plan(c.key, day, round(body.origin.lat, 3), round(body.origin.lon, 3),
                               round(body.dest.lat, 3), round(body.dest.lon, 3), body.ice_class, body.cruise_kn,
                               body.w_time, body.w_risk, body.avoid_bergs, body.w_ice, body.berg_margin_nm)


@app.post("/api/route")
def route(body: RouteBody):
    c, day, plan = _plan(body)
    out = dict(plan)
    _, stats = sync_bundle(c, day, body.ice_class, plan)
    out["sync"] = stats
    return out


@app.post("/api/sync/bundle")
def bundle(body: RouteBody):
    c, day, plan = _plan(body)
    data, _ = sync_bundle(c, day, body.ice_class, plan)
    return Response(data, media_type="application/octet-stream",
                    headers={"Content-Disposition": f'attachment; filename="polarnav_{C.index_day(day)}.pnb"'})


@app.get("/api/sar")
def sar_scene(seed: int = Query(7, ge=0, le=10_000), pfa: float = Query(1e-5, gt=0, lt=0.1)):
    return sar.run(seed, pfa)


# ------------------------------------------------------------------ dashboard
if C.FRONTEND_DIST.exists():
    app.mount("/assets", StaticFiles(directory=C.FRONTEND_DIST / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str):
        f = C.FRONTEND_DIST / path
        if path and f.is_file():
            return FileResponse(f)
        return FileResponse(C.FRONTEND_DIST / "index.html")

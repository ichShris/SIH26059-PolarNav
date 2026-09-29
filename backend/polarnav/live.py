"""Live data mode: real observations and forecasts in place of the synthetic twin.

Sources (all free, no account needed):
  * Sea-ice concentration — NSIDC Sea Ice Index (G02135) v4 daily GeoTIFF, 25 km,
    NSIDC polar stereographic south (EPSG:3412), ~1 day latency.
  * 10 m winds — Open-Meteo, ECMWF IFS 0.25 deg: the past days (analysis / short-range)
    plus the 4-day forecast, sampled on a 500 km grid and interpolated.
  * Surface currents — Open-Meteo Marine API; the synthetic climatology fills gaps
    (e.g. under sea ice, where the marine model returns nothing).
  * Icebergs — US National Ice Center Antarctic iceberg list (CSV, ~weekly).

LiveStore/LiveWorld implement the same interface as DataStore/World, so the ConvLSTM,
drift engine, POLARIS and router run unchanged on live inputs. What remains modelled:
ice thickness (diagnostic from concentration — no free daily thickness product), iceberg
keel depth (USNIC gives length and width only), and wind ensemble spread (perturbations
around the deterministic ECMWF run).
"""
from __future__ import annotations

import csv
import datetime as dt
import io
import json
import threading
import time
import urllib.error
import urllib.request

import numpy as np
from scipy import ndimage

from . import config as C
from .data import DataStore
from .projection import en_to_xy, to_latlon

CACHE = C.ROOT / "backend" / "cache" / "live"
UA = {"User-Agent": "PolarNav/0.1 (SIH 2026 prototype)"}
NSIDC_BASE = "https://noaadata.apps.nsidc.org/NOAA/G02135/south/daily/geotiff"
OPEN_METEO = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_MARINE = "https://marine-api.open-meteo.com/v1/marine"
USNIC_CSV = "https://usicecenter.gov/File/DownloadCurrent?pId=134"

WIND_HALF_KM, WIND_STEP_KM = 5000.0, 500.0   # 21 x 21 sample points
CUR_HALF_KM, CUR_STEP_KM = 5000.0, 1000.0    # 11 x 11 sample points
PAST_DAYS, FORECAST_DAYS = 12, 4
BERG_HEIGHT_M = 250.0                         # typical tabular berg thickness (not reported by USNIC)


# ---------------------------------------------------------------- HTTP with disk cache
def _get(url: str, timeout: int = 60, retries: int = 2) -> bytes:
    last = None
    for k in range(retries + 1):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
                return r.read()
        except (urllib.error.URLError, TimeoutError) as e:
            last = e
            if isinstance(e, urllib.error.HTTPError) and e.code == 404:
                raise
            time.sleep(1.5 * (k + 1))
    raise RuntimeError(f"download failed: {url} ({last})")


def _cached(name: str, url: str, max_age_s: float | None) -> bytes:
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / name
    if f.exists() and (max_age_s is None or time.time() - f.stat().st_mtime < max_age_s):
        return f.read_bytes()
    data = _get(url)
    f.write_bytes(data)
    return data


# ---------------------------------------------------------------- NSIDC sea ice
_A, _E2 = 6378273.0, 0.006693883  # Hughes 1980 ellipsoid (EPSG:3412)
_E = np.sqrt(_E2)


def _t(phi):
    return np.tan(np.pi / 4 - phi / 2) / ((1 - _E * np.sin(phi)) / (1 + _E * np.sin(phi))) ** (_E / 2)


_PC = np.deg2rad(70.0)
_MC = np.cos(_PC) / np.sqrt(1 - _E2 * np.sin(_PC) ** 2)
_TC = _t(_PC)


def nsidc_xy(lat, lon):
    """(lat, lon) -> NSIDC south polar stereographic km (EPSG:3412, true scale 70 S)."""
    phi = np.deg2rad(-np.asarray(lat, float))
    lam = np.deg2rad(np.asarray(lon, float))
    rho = _A * _MC * _t(phi) / _TC
    return rho * np.sin(lam) / 1000.0, rho * np.cos(lam) / 1000.0


def _nsidc_url(d: dt.date, version: str) -> str:
    return f"{NSIDC_BASE}/{d:%Y}/{d:%m}_{d:%b}/S_{d:%Y%m%d}_concentration_{version}.tif"


def latest_nsidc_date(today: dt.date | None = None) -> dt.date:
    """Most recent daily file on the NSIDC server (usually yesterday)."""
    today = today or dt.datetime.now(dt.timezone.utc).date()
    for back in range(0, 40):
        d = today - dt.timedelta(days=back)
        try:
            _nsidc_tif(d)
            return d
        except (urllib.error.HTTPError, RuntimeError):
            continue
    raise RuntimeError("no NSIDC sea-ice file found in the last 40 days")


def _nsidc_tif(d: dt.date) -> bytes:
    for v in ("v4.0", "v3.0"):
        name = f"nsidc_S_{d:%Y%m%d}_{v}.tif"
        if (CACHE / name).exists():
            return (CACHE / name).read_bytes()
    last = None
    for v in ("v4.0", "v3.0"):
        try:
            return _cached(f"nsidc_S_{d:%Y%m%d}_{v}.tif", _nsidc_url(d, v), None)
        except urllib.error.HTTPError as e:
            last = e
    raise last or RuntimeError("NSIDC download failed")


class _Regrid:
    """Area-average NSIDC 25 km cells onto the 50 km model grid (2 x 2 sub-samples per cell)."""

    def __init__(self):
        n, h = C.ICE_N, C.ICE_HALF_KM
        off = np.array([-12.5, 12.5])
        xs = -h + C.CELL_KM * (np.arange(n) + 0.5)
        X, Y = np.meshgrid(xs, xs)  # row 0 = south
        subs = []
        for dx in off:
            for dy in off:
                lat, lon = to_latlon(X + dx, Y + dy)
                nx, ny = nsidc_xy(lat, lon)
                col = np.floor((nx + 3950.0) / 25.0).astype(int)
                row = np.floor((4350.0 - ny) / 25.0).astype(int)
                subs.append((np.clip(row, 0, 331), np.clip(col, 0, 315), (col >= 0) & (col < 316) & (row >= 0) & (row < 332)))
        self.subs = subs

    def __call__(self, raw: np.ndarray) -> np.ndarray:
        acc = np.zeros((C.ICE_N, C.ICE_N))
        cnt = np.zeros((C.ICE_N, C.ICE_N))
        for r, c, ok in self.subs:
            v = raw[r, c].astype(float)
            valid = ok & (v <= 1000)          # 0-1000 = SIC x10; 2510+ = pole hole, coast, land, missing
            acc += np.where(valid, v / 1000.0, 0.0)
            cnt += valid
        out = np.where(cnt > 0, acc / np.maximum(cnt, 1), 0.0)
        return out.astype(np.float32)


# ---------------------------------------------------------------- Open-Meteo
def _grid_points(half: float, step: float):
    ax = np.arange(-half, half + 1e-6, step)
    X, Y = np.meshgrid(ax, ax)  # row 0 = south
    lat, lon = to_latlon(X, Y)
    return ax, lat, lon


def _om_query(base: str, lat, lon, params: dict) -> list[dict]:
    q = "&".join(f"{k}={v}" for k, v in params.items())
    url = (f"{base}?latitude={','.join(f'{a:.3f}' for a in lat.ravel())}"
           f"&longitude={','.join(f'{o:.3f}' for o in lon.ravel())}&{q}")
    return url


class LiveWinds:
    """Hourly 10 m wind on a 500 km grid (map x/y components, m/s), with space-time interpolation."""

    def __init__(self, hour_key: str):
        ax, lat, lon = _grid_points(WIND_HALF_KM, WIND_STEP_KM)
        params = {"hourly": "wind_speed_10m,wind_direction_10m", "wind_speed_unit": "ms", "models": "ecmwf_ifs025",
                  "past_days": PAST_DAYS, "forecast_days": FORECAST_DAYS, "timezone": "UTC"}
        raw = json.loads(_cached(f"om_wind_{hour_key}.json", _om_query(OPEN_METEO, lat, lon, params), 3600))
        raw = raw if isinstance(raw, list) else [raw]
        n = len(ax)
        times = raw[0]["hourly"]["time"]
        t0 = dt.datetime.fromisoformat(times[0])
        self.t0_h = (t0.date() - C.SIM_START).days * 24.0 + t0.hour
        T = len(times)
        spd = np.array([[np.nan if v is None else v for v in r["hourly"]["wind_speed_10m"]] for r in raw], float)
        ddir = np.array([[np.nan if v is None else v for v in r["hourly"]["wind_direction_10m"]] for r in raw], float)
        # fill occasional gaps along time
        for a in (spd, ddir):
            for i in range(a.shape[0]):
                bad = np.isnan(a[i])
                if bad.all():
                    a[i] = 0.0
                elif bad.any():
                    a[i, bad] = np.interp(np.flatnonzero(bad), np.flatnonzero(~bad), a[i, ~bad])
        # meteorological direction = where the wind comes FROM
        ue = -spd * np.sin(np.deg2rad(ddir))
        vn = -spd * np.cos(np.deg2rad(ddir))
        ux, vy = en_to_xy(ue, vn, lon.ravel()[:, None])
        self.u = ux.T.reshape(T, n, n).astype(np.float32)
        self.v = vy.T.reshape(T, n, n).astype(np.float32)
        self.ax0, self.step, self.n, self.T = ax[0], WIND_STEP_KM, n, T
        self.model_run = raw[0].get("generationtime_ms")
        self.first_time, self.last_time = times[0], times[-1]

    def at(self, x, y, t_h):
        """Bilinear in space, linear in time; clamps outside the sampled domain/period."""
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        f = np.clip(t_h - self.t0_h, 0, self.T - 1)
        k = int(min(np.floor(f), self.T - 2))
        w = f - k
        coords = [np.atleast_1d((y - self.ax0) / self.step), np.atleast_1d((x - self.ax0) / self.step)]
        out = []
        for fld in (self.u, self.v):
            a = ndimage.map_coordinates(fld[k], coords, order=1, mode="nearest")
            b = ndimage.map_coordinates(fld[k + 1], coords, order=1, mode="nearest")
            out.append(((1 - w) * a + w * b).reshape(x.shape))
        return out[0], out[1]


def live_currents(hour_key: str, base_u: np.ndarray, base_v: np.ndarray, blocked: np.ndarray):
    """Nav-grid surface currents: Open-Meteo Marine where available, synthetic climatology elsewhere."""
    ax, lat, lon = _grid_points(CUR_HALF_KM, CUR_STEP_KM)
    X, Y = np.meshgrid(ax, ax)
    # the Marine API rejects the whole request if any point is on land, so ask for ocean points only
    ci = np.clip(((X + C.NAV_HALF_KM) // C.CELL_KM).astype(int), 0, C.NAV_N - 1)
    ri = np.clip(((Y + C.NAV_HALF_KM) // C.CELL_KM).astype(int), 0, C.NAV_N - 1)
    sea = ~ndimage.binary_dilation(blocked, iterations=2)[ri, ci]
    idx = np.flatnonzero(sea.ravel())
    params = {"hourly": "ocean_current_velocity,ocean_current_direction", "forecast_days": 1, "timezone": "UTC"}
    try:
        raw = json.loads(_cached(f"om_cur_{hour_key}.json",
                                 _om_query(OPEN_METEO_MARINE, lat.ravel()[idx], lon.ravel()[idx], params), 3 * 3600))
    except Exception as e:
        print(f"live: marine currents unavailable ({e}); using climatology")
        return base_u, base_v, 0
    raw = raw if isinstance(raw, list) else [raw]
    n = len(ax)
    U = np.full((n, n), np.nan)
    V = np.full((n, n), np.nan)
    for i, r in zip(idx, raw):
        vel = [v for v in r.get("hourly", {}).get("ocean_current_velocity", []) if v is not None]
        dirs = [v for v in r.get("hourly", {}).get("ocean_current_direction", []) if v is not None]
        if not vel or not dirs:
            continue
        s = np.mean(vel) / 3.6  # km/h -> m/s
        d = np.deg2rad(np.mean(dirs))  # oceanographic: direction the current flows TOWARD
        ux, vy = en_to_xy(s * np.sin(d), s * np.cos(d), lon.ravel()[i])
        U.flat[i], V.flat[i] = ux, vy
    have = ~np.isnan(U)
    if have.sum() < 4:
        return base_u, base_v, 0
    # interpolate the sampled currents onto the nav grid, then blend: live where sampled, climatology elsewhere
    col = (C.NAV_X - ax[0]) / CUR_STEP_KM
    rowc = (C.NAV_Y - ax[0]) / CUR_STEP_KM
    R, Cc = np.meshgrid(rowc, col, indexing="ij")
    wgt = ndimage.map_coordinates(have.astype(float), [R, Cc], order=1, mode="constant", cval=0.0)
    lu = ndimage.map_coordinates(np.nan_to_num(U), [R, Cc], order=1, mode="constant", cval=0.0)
    lv = ndimage.map_coordinates(np.nan_to_num(V), [R, Cc], order=1, mode="constant", cval=0.0)
    lu = np.where(wgt > 1e-6, lu / np.maximum(wgt, 1e-6), 0.0)
    lv = np.where(wgt > 1e-6, lv / np.maximum(wgt, 1e-6), 0.0)
    w = np.clip(wgt, 0, 1)
    cu = (w * lu + (1 - w) * base_u).astype(np.float32)
    cv = (w * lv + (1 - w) * base_v).astype(np.float32)
    cu[blocked] = 0.0
    cv[blocked] = 0.0
    return cu, cv, int(have.sum())


# ---------------------------------------------------------------- icebergs
def usnic_icebergs() -> dict:
    raw = _cached("usnic_icebergs.csv", USNIC_CSV, 6 * 3600).decode("utf-8-sig")
    bergs = []
    report = None
    for row in csv.DictReader(io.StringIO(raw)):
        try:
            d = dt.datetime.strptime(row["Last Update"].strip(), "%m/%d/%Y").date()
            bergs.append({"name": row["Iceberg"].strip(), "lat": float(row["Latitude"]), "lon": float(row["Longitude"]),
                          "length_m": float(row["Length (NM)"]) * 1852.0, "width_m": float(row["Width (NM)"]) * 1852.0,
                          "reported": d})
            report = max(report, d) if report else d
        except (KeyError, ValueError):
            continue
    return {"bergs": bergs, "report_date": report}


# ---------------------------------------------------------------- live world / store
class LiveWorld:
    """World-compatible facade: live winds and currents, synthetic geography and thickness diagnostic."""

    def __init__(self, base, winds: LiveWinds, cur_u: np.ndarray, cur_v: np.ndarray):
        self.base = base
        self.winds = winds
        self.cur_u, self.cur_v = cur_u, cur_v
        for a in ("lat", "lon", "gx", "gy", "ice_sl", "ice_lat", "ice_lon", "ice_blocked", "ice_coast", "blocked", "coast_km"):
            setattr(self, a, getattr(base, a))

    def wind(self, x, y, t_h, issue_h=None, member=0):
        return self.winds.at(x, y, t_h)

    def wind_particles(self, x, y, t_h, issue_h, members):
        u, v = self.winds.at(x, y, t_h)
        if issue_h is None:
            return u, v
        # ensemble spread around the deterministic ECMWF run: per-member rotation, gain and bias
        # that grow with lead time (member 0 is the control)
        lead = max(t_h - issue_h, 0.0) / 72.0
        m = np.asarray(members)
        rng = np.random.default_rng(int(issue_h))
        draws = rng.standard_normal((int(m.max()) + 1, 4))
        draws[0] = 0.0
        d = draws[m]
        ang = np.deg2rad(12.0) * d[:, 0] * lead
        gain = 1.0 + 0.15 * d[:, 1] * lead
        ca, sa = np.cos(ang), np.sin(ang)
        uu = gain * (ca * u - sa * v) + 2.0 * d[:, 2] * lead
        vv = gain * (sa * u + ca * v) + 2.0 * d[:, 3] * lead
        return uu, vv

    def current(self, x, y):
        from .synth import _bilinear
        x = np.asarray(x, dtype=float)
        return _bilinear(self.cur_u, x, y).reshape(x.shape), _bilinear(self.cur_v, x, y).reshape(x.shape)

    def thickness(self, sic, day):
        return self.base.thickness(sic, day)


class LiveStore:
    """DataStore-compatible view of the live feeds for the latest available sea-ice date."""

    def __init__(self, sim: DataStore, log=print):
        t0 = time.time()
        self.sim = sim
        self.masks = sim.masks
        self.ice_blocked, self.ice_coast = sim.ice_blocked, sim.ice_coast
        self.ice_lat, self.ice_lon = sim.ice_lat, sim.ice_lon
        now = dt.datetime.now(dt.timezone.utc)
        self.fetched_at = now.isoformat(timespec="seconds")
        hour_key = now.strftime("%Y%m%d%H")
        self.analysis_date = latest_nsidc_date(now.date())
        self.day0 = C.day_index(self.analysis_date)
        self.n_days = self.day0 + C.FORECAST_DAYS + 1
        self._regrid = _Regrid()
        self._sic: dict[int, np.ndarray] = {}
        log(f"live: NSIDC analysis {self.analysis_date}")
        self.winds = LiveWinds(hour_key)
        log(f"live: ECMWF IFS winds {self.winds.first_time} .. {self.winds.last_time}")
        cu, cv, n_cur = live_currents(hour_key, sim.world.cur_u, sim.world.cur_v, sim.masks["blocked"])
        self.n_current_points = n_cur
        self.world = LiveWorld(sim.world, self.winds, cu, cv)
        self.icebergs = usnic_icebergs()
        log(f"live: {len(self.icebergs['bergs'])} USNIC icebergs (report {self.icebergs['report_date']}); "
            f"{n_cur} live current points; ready in {time.time() - t0:.1f}s")
        xs, ys = sim.world.gx[sim.world.ice_sl][::2, ::2], sim.world.gy[sim.world.ice_sl][::2, ::2]
        self._coarse_xy = (xs, ys)

    # --- sea ice
    def sic_obs(self, day: int) -> np.ndarray:
        if day > self.day0:
            raise KeyError("no observation yet for a future day")
        if day not in self._sic:
            import tifffile
            d = C.index_day(day)
            try:
                raw = tifffile.imread(io.BytesIO(_nsidc_tif(d)))
                s = self._regrid(raw)
            except Exception:
                # a missing daily file: fall back to the nearest earlier day (persistence)
                s = self.sic_obs(day - 1)
            s[self.ice_blocked] = 0.0
            self._sic[day] = s
        return self._sic[day]

    def sic_truth(self, day: int) -> np.ndarray:
        """Verifying 'truth' in live mode is the satellite analysis itself (only for past days)."""
        return self.sic_obs(day)

    # --- winds, daily means on the coarse (100 km) ice grid
    def _daily_wind(self, day: int) -> np.ndarray:
        xs, ys = self._coarse_xy
        acc = np.zeros((2,) + xs.shape)
        for hh in (3.0, 9.0, 15.0, 21.0):
            u, v = self.winds.at(xs, ys, day * 24.0 + hh)
            acc[0] += u
            acc[1] += v
        return (acc / 4.0).astype(np.float32)

    def wind_truth(self, day: int, coarse: bool = True) -> np.ndarray:
        from .synth import upsample2
        w = self._daily_wind(day)
        return w if coarse else upsample2(w)

    def wind_forecast(self, issue_day: int, coarse: bool = True) -> np.ndarray:
        """Winds for issue+1..+3. For past issue days this is the analysed wind (archived
        forecasts are not fetched), which slightly flatters the real-time check."""
        from .synth import upsample2
        w = np.stack([self._daily_wind(issue_day + 1 + k) for k in range(C.FORECAST_DAYS)])
        return w if coarse else upsample2(w)

    def thickness(self, sic, day):
        return self.sim.world.thickness(sic, day)

    embed = staticmethod(DataStore.embed)

    def sources(self) -> dict:
        return {
            "sea_ice": {"name": "NSIDC Sea Ice Index G02135 v4 (daily, 25 km)", "valid": str(self.analysis_date)},
            "wind": {"name": "ECMWF IFS 0.25° via Open-Meteo", "from": self.winds.first_time, "to": self.winds.last_time},
            "currents": {"name": "Open-Meteo Marine (gaps filled with climatology)", "points": self.n_current_points},
            "icebergs": {"name": "US National Ice Center Antarctic icebergs", "report": str(self.icebergs["report_date"]),
                         "count": len(self.icebergs["bergs"])},
            "modelled": ["ice thickness (diagnostic from concentration)", "iceberg keel depth (250 m assumed)",
                         "wind ensemble spread (perturbations around the ECMWF run)"],
            "fetched_at": self.fetched_at,
        }


_live: LiveStore | None = None
_live_lock = threading.Lock()
LIVE_TTL_S = 3600


def get_live(sim: DataStore, refresh: bool = False, log=print) -> LiveStore:
    """Cached LiveStore, rebuilt at most hourly (or on request)."""
    global _live
    with _live_lock:
        stale = _live is None or refresh or (
            time.time() - dt.datetime.fromisoformat(_live.fetched_at).timestamp() > LIVE_TTL_S)
        if stale:
            _live = LiveStore(sim, log=log)
        return _live

"""Synthetic but physically structured Southern Ocean environment.

The prototype has to run offline without Copernicus/ECMWF/NSIDC credentials, so it
ships with a reproducible "digital twin" of the ocean that stands in for the real
feeds. Every downstream module (ConvLSTM, drift model, POLARIS, router) consumes the
same arrays a real ingestion pipeline would produce, so swapping in ERA5 winds, OSI-SAF
/ NSIDC sea-ice concentration and CMEMS currents only touches ``data.py``.

What is modelled:
* Winds (ERA5 stand-in): mid-latitude westerlies, coastal easterlies, katabatic
  offshore flow, plus a catalogue of transient cyclones that orbit the continent
  eastward and spin clockwise (Southern Hemisphere lows) with frictional inflow.
* Forecast winds (ECMWF HRES/ENS stand-in): the same cyclones with position and
  intensity errors that grow with lead time; ensemble members draw different errors.
* Currents (CMEMS stand-in): Antarctic Circumpolar Current, westward Antarctic Coastal
  Current, Weddell and Ross gyres.
* Sea ice: a seasonal climatology (fast Oct-Feb retreat, slow Mar-Sep advance, with
  Weddell/Ross/Amundsen summer remnants and coastal polynyas) plus a daily dynamic
  state advected by free-drift ice velocity (2% of wind, 30 deg left of it, plus
  current), with divergence opening leads and stochastic anomalies.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from . import config as C
from .projection import en_to_xy, to_latlon, to_xy

HOURS_PER_DAY = 24.0


def _angdiff(a, b):
    return (np.asarray(a) - b + 180.0) % 360.0 - 180.0


def upsample2(a: np.ndarray) -> np.ndarray:
    """Bilinear x2 upsampling over the last two axes (coarse wind grid -> ice grid)."""
    zoom = [1.0] * (a.ndim - 2) + [2.0, 2.0]
    return ndimage.zoom(np.asarray(a, dtype=np.float32), zoom, order=1, mode="nearest", grid_mode=True)


def _bilinear(field: np.ndarray, x, y) -> np.ndarray:
    """Sample a nav-grid field at arbitrary projected km positions."""
    col = (np.asarray(x) - C.NAV_X[0]) / C.CELL_KM
    row = (np.asarray(y) - C.NAV_Y[0]) / C.CELL_KM
    return ndimage.map_coordinates(field, [np.atleast_1d(row), np.atleast_1d(col)], order=1, mode="nearest")


class World:
    """Holds static geography and generates winds, currents and sea ice."""

    N_CYCLONES_PER_DAY = 1.3

    def __init__(self, masks: dict[str, np.ndarray], seed: int = 2026):
        self.seed = seed
        self._err_cache: dict = {}
        self.blocked = masks["blocked"]
        self.coast_km = masks["coast_km"]
        gx, gy = np.meshgrid(C.NAV_X, C.NAV_Y)
        self.gx, self.gy = gx, gy
        self.lat, self.lon = to_latlon(gx, gy)
        sl = slice(C.ICE_OFF, C.ICE_OFF + C.ICE_N)
        self.ice_sl = (sl, sl)
        self.ice_blocked = self.blocked[self.ice_sl]
        self.ice_coast = self.coast_km[self.ice_sl]
        self.ice_lat, self.ice_lon = self.lat[self.ice_sl], self.lon[self.ice_sl]
        self._build_background_wind()
        self._build_currents()
        self._build_cyclones()
        self._build_ice_climatology()

    # ------------------------------------------------------------------ winds
    def _build_background_wind(self):
        lat = self.lat
        u_e = 11.0 * np.exp(-((lat + 50.0) / 9.0) ** 2) - 6.0 * np.exp(-((lat + 67.0) / 4.0) ** 2)
        v_n = 1.0 * np.exp(-((lat + 55.0) / 8.0) ** 2)
        bx, by = en_to_xy(u_e, v_n, self.lon)
        # katabatic outflow: down the gradient of distance-to-coast, decaying offshore
        cs = ndimage.gaussian_filter(self.coast_km, 3.0)
        gy_, gx_ = np.gradient(cs)
        norm = np.hypot(gx_, gy_) + 1e-6
        kat = 5.0 * np.exp(-self.coast_km / 180.0)
        bx = bx + kat * gx_ / norm
        by = by + kat * gy_ / norm
        self.bg_u = bx.astype(np.float32)
        self.bg_v = by.astype(np.float32)

    def _build_cyclones(self):
        rng = np.random.default_rng(self.seed)
        span_h = (C.N_DAYS + 10) * HOURS_PER_DAY
        n = int(self.N_CYCLONES_PER_DAY * (C.N_DAYS + 20))
        tb = rng.uniform(-8 * HOURS_PER_DAY, span_h, n)
        cyc = np.empty((n, 9))
        cyc[:, 0] = tb
        cyc[:, 1] = rng.uniform(60, 150, n)            # lifetime h
        cyc[:, 2] = rng.uniform(-180, 180, n)          # genesis lon
        cyc[:, 3] = rng.uniform(-66, -50, n)           # genesis lat
        cyc[:, 4] = rng.uniform(0.35, 0.8, n)          # eastward deg/h of longitude
        cyc[:, 5] = rng.uniform(-0.06, -0.01, n)       # poleward drift deg/h
        cyc[:, 6] = rng.uniform(10, 24, n)             # peak wind m/s
        cyc[:, 7] = rng.uniform(350, 850, n)           # radius of max wind km
        cyc[:, 8] = np.arange(n)
        self.cyclones = cyc[np.argsort(tb)]

    def _cyclone_base(self, t_h: float):
        c = self.cyclones
        age = t_h - c[:, 0]
        live = (age >= 0) & (age <= c[:, 1])
        c, age = c[live], age[live]
        lon = c[:, 2] + c[:, 4] * age
        lat = np.clip(c[:, 3] + c[:, 5] * age, -72, -45)
        amp = c[:, 6] * np.sin(np.pi * age / c[:, 1]) ** 0.7
        cx, cy = to_xy(lat, lon)
        return cx, cy, amp, c[:, 7], c[:, 8]

    def _perturb(self, base, t_h: float, issue_h: float, member: int):
        cx, cy, amp, rm, ids = base
        lead = max(t_h - issue_h, 0.0)
        # deterministic per (issue, member, cyclone) error draws
        e = np.array([self._cyc_error(int(issue_h), member, int(i)) for i in ids]).reshape(-1, 3)
        sig = 4.2 * lead  # km; ~100 km at 24 h, ~300 km at 72 h
        return (cx + e[:, 0] * sig, cy + e[:, 1] * sig,
                amp * (1.0 + 0.12 * e[:, 2] * min(lead / 72.0, 1.5)), rm)

    def _cyclone_state(self, t_h: float, issue_h: float | None, member: int):
        base = self._cyclone_base(t_h)
        if issue_h is None:
            return base[:4]
        return self._perturb(base, t_h, issue_h, member)

    def _cyc_error(self, issue_h: int, member: int, cyc_id: int) -> np.ndarray:
        key = (issue_h, member, cyc_id)
        e = self._err_cache.get(key)
        if e is None:
            if len(self._err_cache) > 200_000:
                self._err_cache.clear()
            e = self._err_cache[key] = np.random.default_rng([issue_h, member, cyc_id]).standard_normal(3)
        return e

    def wind_particles(self, x, y, t_h: float, issue_h: float | None, members: np.ndarray):
        """Wind at particle positions where each particle belongs to an ensemble member."""
        if issue_h is None:
            return self.wind(x, y, t_h)
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        u = _bilinear(self.bg_u, x, y).reshape(x.shape)
        v = _bilinear(self.bg_v, x, y).reshape(x.shape)
        uniq, inv = np.unique(members, return_inverse=True)
        base = self._cyclone_base(t_h)
        st = [self._perturb(base, t_h, issue_h, int(m)) for m in uniq]
        if len(st[0][0]):
            cx = np.stack([s[0] for s in st])[inv]
            cy = np.stack([s[1] for s in st])[inv]
            amp = np.stack([s[2] for s in st])[inv]
            rm = st[0][3][None, :]
            dx = x[:, None] - cx
            dy = y[:, None] - cy
            r = np.hypot(dx, dy) + 1e-3
            q = r / rm
            spd = amp * q * np.exp(0.5 * (1.0 - q * q))
            ca, sa = np.cos(np.deg2rad(20)), np.sin(np.deg2rad(20))
            u = u + (spd * (ca * dy - sa * dx) / r).sum(-1)
            v = v + (spd * (-ca * dx - sa * dy) / r).sum(-1)
        return u, v

    def wind(self, x, y, t_h: float, issue_h: float | None = None, member: int = 0):
        """10 m wind (m/s, map x/y components) at projected positions and hour t_h.

        issue_h=None returns the verifying "truth" (reanalysis); otherwise a forecast
        issued at issue_h, with error that grows with lead time. member>0 gives
        ensemble members with independent errors.
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        u = _bilinear(self.bg_u, x, y).reshape(x.shape)
        v = _bilinear(self.bg_v, x, y).reshape(x.shape)
        cx, cy, amp, rm = self._cyclone_state(t_h, issue_h, member)
        if len(cx):
            dx = x[..., None] - cx
            dy = y[..., None] - cy
            r = np.hypot(dx, dy) + 1e-3
            q = r / rm
            spd = amp * q * np.exp(0.5 * (1.0 - q * q))
            ca, sa = np.cos(np.deg2rad(20)), np.sin(np.deg2rad(20))
            # clockwise tangential (dy, -dx)/r plus inflow -(dx, dy)/r
            u = u + (spd * (ca * dy - sa * dx) / r).sum(-1)
            v = v + (spd * (-ca * dx - sa * dy) / r).sum(-1)
        return u, v

    def wind_grid_ice(self, day: int, issue_day: int | None = None, member: int = 0, coarse: bool = False):
        """Daily-mean wind on the ice grid, shape (2, N, N) (or (2, N/2, N/2) if coarse).

        Winds are evaluated on a 100 km sub-grid (synoptic features are 350-850 km
        across) and bilinearly upsampled.
        """
        xs, ys = self.gx[self.ice_sl][::2, ::2], self.gy[self.ice_sl][::2, ::2]
        issue_h = None if issue_day is None else issue_day * HOURS_PER_DAY
        acc_u = np.zeros_like(xs)
        acc_v = np.zeros_like(xs)
        for hh in (3.0, 9.0, 15.0, 21.0):
            u, v = self.wind(xs, ys, day * HOURS_PER_DAY + hh, issue_h, member)
            acc_u += u
            acc_v += v
        w = np.stack([acc_u, acc_v]) / 4.0
        return w if coarse else upsample2(w)

    # ------------------------------------------------------------------ currents
    def _build_currents(self):
        lat, lon = self.lat, self.lon
        u_e = 0.22 * np.exp(-((lat + 54.0) / 6.0) ** 2) - 0.14 * np.exp(-self.coast_km / 160.0)
        cx, cy = en_to_xy(u_e, np.zeros_like(u_e), lon)
        for glat, glon, spd, rad in ((-64.0, -30.0, 0.09, 800.0), (-70.0, -160.0, 0.07, 650.0)):
            gx0, gy0 = to_xy(glat, glon)
            dx, dy = self.gx - gx0, self.gy - gy0
            r = np.hypot(dx, dy) + 1e-3
            q = r / rad
            s = spd * q * np.exp(0.5 * (1 - q * q))
            cx += s * dy / r
            cy += s * -dx / r
        cx[self.blocked] = 0.0
        cy[self.blocked] = 0.0
        self.cur_u = cx.astype(np.float32)
        self.cur_v = cy.astype(np.float32)

    def current(self, x, y):
        x = np.asarray(x, dtype=float)
        return _bilinear(self.cur_u, x, y).reshape(x.shape), _bilinear(self.cur_v, x, y).reshape(x.shape)

    # ------------------------------------------------------------------ sea ice
    def _build_ice_climatology(self):
        lon = self.ice_lon
        self.d_winter = (900 + 750 * np.exp(-(_angdiff(lon, -25) / 35) ** 2)
                         + 420 * np.exp(-(_angdiff(lon, -170) / 30) ** 2)
                         - 450 * np.exp(-(_angdiff(lon, -62) / 10) ** 2))
        self.d_summer = (30 + 560 * np.exp(-(_angdiff(lon, -48) / 14) ** 2)
                         + 240 * np.exp(-(_angdiff(lon, -105) / 25) ** 2)
                         + 160 * np.exp(-(_angdiff(lon, -155) / 14) ** 2))
        poly = np.zeros_like(lon)
        for plon, strength in ((76.0, 0.75), (69.0, 0.5), (175.0, 0.6), (146.0, 0.5), (-20.0, 0.35), (12.0, 0.3)):
            poly = np.maximum(poly, strength * np.exp(-(_angdiff(lon, plon) / 5.0) ** 2))
        self.polynya = poly * np.exp(-self.ice_coast / 90.0)
        rng = np.random.default_rng(self.seed + 1)
        self.age_noise = ndimage.gaussian_filter(rng.standard_normal(lon.shape), 4.0)
        self.age_noise /= self.age_noise.std() + 1e-9

    @staticmethod
    def season(doy: float) -> float:
        """0 at the late-February minimum, 1 at the late-September maximum."""
        lo, hi = 52.0, 265.0
        d = doy if doy >= lo else doy + 365.0
        if d <= hi:
            return 0.5 - 0.5 * np.cos(np.pi * (d - lo) / (hi - lo))
        return 0.5 + 0.5 * np.cos(np.pi * (d - hi) / (lo + 365.0 - hi))

    def ice_edge_km(self, day: int) -> np.ndarray:
        s = self.season(C.index_day(day).timetuple().tm_yday)
        return self.d_summer + s * (self.d_winter - self.d_summer)

    def sic_target(self, day: int) -> np.ndarray:
        edge = self.ice_edge_km(day)
        d = self.ice_coast
        inside = edge - d
        t = np.where(inside > 0, 0.15 + 0.83 * (1 - np.exp(-inside / 120.0)), 0.15 * np.exp(inside / 60.0))
        t = t * (1.0 - self.polynya)
        t = t / (1.0 + np.exp((self.ice_lat + 56.0) / 1.2))  # no ice equatorward of ~56 S
        t[self.ice_blocked] = 0.0
        return np.clip(t, 0, 1)

    def ice_velocity(self, wind_uv: np.ndarray, cur_uv: np.ndarray) -> np.ndarray:
        """Free-drift: 2% of wind turned 30 deg to the left (SH) plus the current, m/s."""
        a = np.deg2rad(30.0)
        wu, wv = wind_uv
        iu = 0.02 * (np.cos(a) * wu - np.sin(a) * wv) + cur_uv[0]
        iv = 0.02 * (np.sin(a) * wu + np.cos(a) * wv) + cur_uv[1]
        return np.stack([iu, iv])

    def step_sic(self, sic: np.ndarray, day: int, wind_uv: np.ndarray, rng) -> np.ndarray:
        cur = np.stack([self.cur_u[self.ice_sl], self.cur_v[self.ice_sl]])
        vel = self.ice_velocity(wind_uv, cur)
        disp = vel * 86400.0 / 1000.0 / C.CELL_KM  # cells per day
        n = sic.shape[0]
        rr, cc = np.mgrid[0:n, 0:n].astype(float)
        adv = ndimage.map_coordinates(sic, [rr - disp[1], cc - disp[0]], order=1, mode="nearest")
        div = np.gradient(disp[0], axis=1) + np.gradient(disp[1], axis=0)
        adv = adv * (1.0 - 0.4 * np.clip(div, -0.5, 0.5))
        tgt = self.sic_target(day)
        tau = 7.0
        new = adv + (tgt - adv) / tau
        noise = ndimage.gaussian_filter(rng.standard_normal((n, n)), 2.5) * 0.35
        new = new + noise * (new > 0.05) * (new < 0.98)
        new[self.ice_blocked] = 0.0
        return np.clip(new, 0.0, 1.0)

    def simulate_sic(self, n_days: int = C.N_DAYS, spinup: int = 60, progress=None):
        rng = np.random.default_rng(self.seed + 2)
        sic = self.sic_target(0)
        for k in range(spinup):  # spin up on repeated day-0 forcing with fresh noise
            sic = self.step_sic(sic, 0, self.wind_grid_ice(0), rng)
        out = np.empty((n_days, C.ICE_N, C.ICE_N), dtype=np.float32)
        winds = np.empty((n_days, 2, C.ICE_N // 2, C.ICE_N // 2), dtype=np.float16)
        for d in range(n_days):
            w = self.wind_grid_ice(d, coarse=True)
            winds[d] = w
            sic = self.step_sic(sic, d, upsample2(w), rng)
            out[d] = sic
            if progress and d % 100 == 0:
                progress(d, n_days)
        return out, winds

    def thickness(self, sic: np.ndarray, day: int) -> np.ndarray:
        """Diagnostic ice thickness (m) from concentration, pack depth, region and season."""
        edge = self.ice_edge_km(day)
        inside = np.clip(edge - self.ice_coast, 0, None)
        s = self.season(C.index_day(day).timetuple().tm_yday)
        base = 0.25 + (0.55 + 0.6 * s) * (1 - np.exp(-inside / 350.0))
        multi_year = 1.9 * np.exp(-(_angdiff(self.ice_lon, -50) / 12) ** 2) * np.exp(-self.ice_coast / 450.0)
        fast = 0.9 * np.exp(-self.ice_coast / 60.0)
        h = (base + multi_year + fast) * (0.35 + 0.65 * sic) * (1.0 + 0.18 * self.age_noise)
        h = np.where(sic < 0.05, 0.0, h)
        return np.clip(h, 0.0, 4.5).astype(np.float32)

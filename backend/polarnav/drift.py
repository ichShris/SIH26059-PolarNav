"""Iceberg drift: physics ensemble + learned residual correction ("hybrid physics-AI").

Physics (Smith 1993; Bigg et al. 1997), per unit mass, map x/y components:

    dv/dt = -f k x v                                   Coriolis
            + (rho_a Ca A_a / 2M) |Va - v| (Va - v)    air drag on the sail
            + (rho_w Cw A_w / 2M) |Vw - v| (Vw - v)    water drag on the keel
            + f k x Vw                                  sea-surface slope (geostrophic current)
            + (v_ice - v) * w(SIC) / tau                capture by dense pack ice

The operational model uses textbook drag coefficients. The synthetic "truth" uses
different ones plus wave-induced (Stokes) drift that the physics omits — exactly the
kind of structural error real drift models have. A complex-valued ridge regression,
fitted on historical tracks, learns a velocity correction from wind/current features;
complex coefficients make the correction rotation-invariant (a scale + turn).
"""
from __future__ import annotations

import json

import numpy as np

from . import config as C
from .data import DataStore
from .projection import scale_factor, to_latlon, to_xy
from .synth import _bilinear

OMEGA = 7.2921e-5
RHO_A, RHO_W, RHO_I = 1.225, 1027.0, 900.0
MODEL_CA, MODEL_CW = 1.3, 0.9          # operational physics
TRUE_CA, TRUE_CW = 1.65, 0.75          # synthetic nature
STOKES_FRAC = 0.012                    # wave drift as a fraction of wind speed (truth only)
DT = 1800.0                            # s
HYBRID_FILE = C.ARTIFACT_DIR / "drift_hybrid.json"
METRICS_FILE = C.ARTIFACT_DIR / "drift_metrics.json"


class Bergs:
    """Array-of-struct container for a set of icebergs (sizes in metres)."""

    def __init__(self, ids, x, y, length, width, height):
        self.ids = list(ids)
        self.x, self.y = np.asarray(x, float), np.asarray(y, float)
        self.length, self.width, self.height = (np.asarray(a, float) for a in (length, width, height))

    def __len__(self):
        return len(self.ids)

    def repeat(self, m: int) -> "Bergs":
        r = lambda a: np.repeat(a, m)
        return Bergs([i for i in self.ids for _ in range(m)], r(self.x), r(self.y),
                     r(self.length), r(self.width), r(self.height))


def _cplx_features(wu, wv, cu, cv, length):
    wa = wu + 1j * wv
    cw = cu + 1j * cv
    small = 1.0 / (1.0 + length / 2000.0)
    return np.stack([wa, np.abs(wa) * wa / 10.0, cw, small * wa], axis=-1)


class DriftEngine:
    def __init__(self, store: DataStore):
        self.s = store
        self.w = store.world
        self.hybrid = None
        self._blocked_f = store.masks["blocked"].astype(np.float32)
        if HYBRID_FILE.exists():
            d = json.loads(HYBRID_FILE.read_text())
            self.hybrid = np.array(d["re"]) + 1j * np.array(d["im"])

    # ------------------------------------------------------------ core integrator
    def simulate(self, bergs: Bergs, t0_h: float, hours: float, *, truth: bool,
                 issue_h: float | None = None, members: np.ndarray | None = None,
                 sic_fn=None, ca=None, cw=None, hybrid: bool = False, record_every_h: float = 1.0):
        """Integrate positions; returns (T, P, 2) km positions at record times, and times."""
        n = len(bergs)
        members = np.zeros(n, int) if members is None else members
        ca = np.full(n, TRUE_CA if truth else MODEL_CA) if ca is None else ca
        cw = np.full(n, TRUE_CW if truth else MODEL_CW) if cw is None else cw
        H, Lg, W = bergs.height, bergs.length, bergs.width
        draft = H * RHO_I / RHO_W
        sail = H - draft
        mass = RHO_I * Lg * W * H
        ka = 0.5 * RHO_A * ca * Lg * sail / mass
        kw = 0.5 * RHO_W * cw * Lg * draft / mass
        x, y = bergs.x.copy(), bergs.y.copy()
        grounded = np.zeros(n, bool)

        def forcing(x, y, t):
            if truth:
                wu, wv = self.w.wind(x, y, t)
            else:
                wu, wv = self.w.wind_particles(x, y, t, issue_h, members)
            cu, cv = self.w.current(x, y)
            return wu, wv, cu, cv

        blocked_f = self._blocked_f

        def step(x, y, vx, vy, t, dt, move=True):
            """Semi-implicit step: exact Coriolis rotation, implicit water drag and ice capture."""
            wu, wv, cu, cv = forcing(x, y, t)
            lat, _ = to_latlon(x, y)
            f = 2 * OMEGA * np.sin(np.deg2rad(lat))
            th = -f * dt  # rotation angle of the inertial oscillation over the step
            c, s_ = np.cos(th), np.sin(th)
            vx, vy = vx * c - vy * s_, vx * s_ + vy * c
            ra = np.hypot(wu - vx, wv - vy)
            kwr = kw * np.hypot(cu - vx, cv - vy)
            num_x = vx + dt * (ka * ra * (wu - vx) - f * cv + kwr * cu)
            num_y = vy + dt * (ka * ra * (wv - vy) + f * cu + kwr * cv)
            den = 1.0 + dt * kwr
            if sic_fn is not None:
                cap = np.clip((sic_fn(x, y, t) - 0.8) / 0.15, 0, 1) / (3 * 3600)
                a = np.deg2rad(30)
                iu = 0.02 * (np.cos(a) * wu - np.sin(a) * wv) + cu
                iv = 0.02 * (np.sin(a) * wu + np.cos(a) * wv) + cv
                num_x = num_x + dt * cap * iu
                num_y = num_y + dt * cap * iv
                den = den + dt * cap
            vx, vy = num_x / den, num_y / den
            if not move:
                return x, y, vx, vy
            ex = np.zeros_like(x)
            ey = np.zeros_like(y)
            if truth:
                small = 1.0 / (1.0 + Lg / 2000.0)
                ex += STOKES_FRAC * small * wu
                ey += STOKES_FRAC * small * wv
            elif hybrid and self.hybrid is not None:
                d = _cplx_features(wu, wv, cu, cv, Lg) @ self.hybrid
                ex += d.real
                ey += d.imag
            k_m = scale_factor(lat) / 1000.0  # metres travelled -> projected km
            return x + dt * (vx + ex) * k_m, y + dt * (vy + ey) * k_m, vx, vy

        # start near local free-drift equilibrium: spin up velocity for 12 h at a fixed position
        vx, vy = np.zeros(n), np.zeros(n)
        for _ in range(12):
            _, _, vx, vy = step(x, y, vx, vy, t0_h, 3600.0, move=False)

        steps = int(round(hours * 3600 / DT))
        rec_every = max(1, int(round(record_every_h * 3600 / DT)))
        out = [np.stack([x, y], -1)]
        t = t0_h
        for k in range(steps):
            nx, ny, vx, vy = step(x, y, vx, vy, t, DT)
            # grounding / land contact: stop the berg
            grounded |= _bilinear(blocked_f, nx, ny) > 0.5
            x = np.where(grounded, x, nx)
            y = np.where(grounded, y, ny)
            vx = np.where(grounded, 0, vx)
            vy = np.where(grounded, 0, vy)
            t += DT / 3600
            if (k + 1) % rec_every == 0:
                out.append(np.stack([x, y], -1))
        return np.stack(out), grounded

    # ------------------------------------------------------------ sea-ice sampling
    def sic_sampler(self, day: int, fields: list[np.ndarray]):
        """fields[k] is the ice-grid SIC valid on day+k; sample by particle time."""
        nav = [self.s.embed(f) for f in fields]

        def fn(x, y, t_h):
            k = int(np.clip(t_h / 24.0 - day, 0, len(nav) - 1))
            return _bilinear(nav[k], x, y).reshape(np.shape(x))
        return fn

    # ------------------------------------------------------------ scenario icebergs
    def seed_bergs(self, day: int, n: int = 34) -> Bergs:
        """Deterministic iceberg population for a scenario date (positions valid 72 h before)."""
        rng = np.random.default_rng(5000 + day)
        blocked = self.s.masks["blocked"]
        coast = self.s.masks["coast_km"]
        lat, lon = self.w.lat, self.w.lon
        ok = (~blocked) & (coast > 30) & (coast < 1100) & (lat < -56)
        # Icebergs concentrate in the coastal current and the marginal ice zone; weight toward
        # the 0-90E sector the Indian expeditions sail through.
        sector = np.exp(-(((lon - 40 + 180) % 360 - 180) / 32.0) ** 2)
        belt = np.exp(-((coast - 380.0) / 260.0) ** 2)
        wgt = ok * belt * (0.12 + sector)
        idx = rng.choice(wgt.size, n, replace=False, p=(wgt / wgt.sum()).ravel())
        j, i = np.unravel_index(idx, wgt.shape)
        x = C.NAV_X[i] + rng.uniform(-20, 20, n)
        y = C.NAV_Y[j] + rng.uniform(-20, 20, n)
        length = np.exp(rng.uniform(np.log(300), np.log(14000), n))
        width = length * rng.uniform(0.35, 0.8, n)
        height = np.clip(45 + 0.02 * length + rng.normal(0, 15, n), 35, 320)
        ids = [f"IB-{int(v):03d}" for v in rng.choice(np.arange(100, 999), n, replace=False)]
        return Bergs(ids, x, y, length, width, height)

    def scenario(self, day: int, sic_forecast: np.ndarray, members: int = 30) -> dict:
        """Observed 72 h history, 72 h ensemble forecast (physics & hybrid), verifying truth."""
        t0 = day * 24.0
        seeds = self.seed_bergs(day)
        truth_fields = [self.s.sic_truth(min(day - 3 + k, self.s.n_days - 1)) for k in range(7)]
        truth_sic = self.sic_sampler(day - 3, truth_fields)
        past, _ = self.simulate(seeds, t0 - 72, 72, truth=True, sic_fn=truth_sic, record_every_h=6)
        now = Bergs(seeds.ids, past[-1, :, 0], past[-1, :, 1], seeds.length, seeds.width, seeds.height)
        fut, grounded_truth = self.simulate(now, t0, 72, truth=True, sic_fn=truth_sic, record_every_h=3)
        fc_sic = self.sic_sampler(day, [self.s.sic_obs(day)] + list(sic_forecast))

        # ensemble: forecast-wind members, drag perturbations, SAR geolocation error
        rng = np.random.default_rng(9000 + day)
        ens = now.repeat(members)
        ens.x = ens.x + rng.normal(0, 0.5, len(ens))
        ens.y = ens.y + rng.normal(0, 0.5, len(ens))
        mem = np.tile(np.arange(members), len(now))
        ca = MODEL_CA * np.exp(rng.normal(0, 0.2, len(ens)))
        cw = MODEL_CW * np.exp(rng.normal(0, 0.2, len(ens)))
        hyb, _ = self.simulate(ens, t0, 72, truth=False, issue_h=t0, members=mem, sic_fn=fc_sic,
                               ca=ca, cw=cw, hybrid=True, record_every_h=3)
        phys, _ = self.simulate(now, t0, 72, truth=False, issue_h=t0, sic_fn=fc_sic, hybrid=False,
                                record_every_h=3)
        hyb = hyb.reshape(hyb.shape[0], len(now), members, 2)
        mean = hyb.mean(2)
        bergs = []
        for b in range(len(now)):
            ell = []
            for k in range(0, hyb.shape[0], 2):  # every 6 h
                pts = hyb[k, b]
                cov = np.cov(pts.T) + np.eye(2) * 0.25
                vals, vecs = np.linalg.eigh(cov)
                ell.append({"t_h": 3 * k, "x": float(mean[k, b, 0]), "y": float(mean[k, b, 1]),
                            "a": float(2.0 * np.sqrt(vals[1])), "b": float(2.0 * np.sqrt(vals[0])),
                            "angle": float(np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1])))})
            lat, lon = to_latlon(now.x[b], now.y[b])
            err_h = float(np.hypot(*(mean[-1, b] - fut[-1, b])))
            err_p = float(np.hypot(*(phys[-1, b] - fut[-1, b])))
            bergs.append({
                "id": now.ids[b], "lat": float(lat), "lon": float(lon),
                "x": float(now.x[b]), "y": float(now.y[b]),
                "length_m": float(now.length[b]), "width_m": float(now.width[b]),
                "height_m": float(now.height[b]),
                "draft_m": float(now.height[b] * RHO_I / RHO_W),
                "observed": np.round(past[:, b], 2).tolist(),
                "forecast": np.round(mean[:, b], 2).tolist(),
                "physics_only": np.round(phys[:, b], 2).tolist(),
                "truth": np.round(fut[:, b], 2).tolist(),
                "members_72h": np.round(hyb[-1, b], 1).tolist(),
                "ellipses": ell,
                "error_72h_km": {"hybrid": err_h, "physics": err_p},
                "grounded": bool(grounded_truth[b]),
            })
        errs_h = [b["error_72h_km"]["hybrid"] for b in bergs]
        errs_p = [b["error_72h_km"]["physics"] for b in bergs]
        return {"step_h": 3, "members": members, "bergs": bergs,
                "summary": {"mean_err_72h_km": {"hybrid": float(np.mean(errs_h)), "physics": float(np.mean(errs_p))}}}

    # ------------------------------------------------------------ hybrid training
    def _windows(self, days: list[int], per_day: int, rng):
        """6 h physics hindcasts restarted from truth; returns features & residual velocities."""
        feats, resid = [], []
        for day in days:
            b = self.seed_bergs(day, per_day)
            b.x += rng.normal(0, 50, len(b))
            truth_sic = self.sic_sampler(day, [self.s.sic_truth(min(day + k, self.s.n_days - 1)) for k in range(4)])
            tr, _ = self.simulate(b, day * 24.0, 72, truth=True, sic_fn=truth_sic, record_every_h=6)
            for k in range(tr.shape[0] - 1):
                t = day * 24.0 + 6 * k
                start = Bergs(b.ids, tr[k, :, 0], tr[k, :, 1], b.length, b.width, b.height)
                md, _ = self.simulate(start, t, 6, truth=False, issue_h=day * 24.0, sic_fn=truth_sic,
                                      record_every_h=6)
                kf = scale_factor(to_latlon(tr[k, :, 0], tr[k, :, 1])[0])
                dv = (tr[k + 1] - md[-1]) * 1000.0 / (6 * 3600) / kf[:, None]  # true m/s
                mid = (tr[k] + tr[k + 1]) / 2
                mx, my = mid[:, 0], mid[:, 1]
                wu, wv = self.w.wind(mx, my, t + 3)  # training uses reanalysis winds
                cu, cv = self.w.current(mx, my)
                feats.append(_cplx_features(wu, wv, cu, cv, b.length))
                resid.append(dv[:, 0] + 1j * dv[:, 1])
        return np.concatenate(feats), np.concatenate(resid)

    def fit_hybrid(self, log=print) -> dict:
        rng = np.random.default_rng(1)
        tr_end = C.day_index(C.TRAIN_END)
        train_days = list(rng.choice(np.arange(10, tr_end - 3), 45, replace=False))
        X, y = self._windows(train_days, 10, rng)
        lam = 1e-3 * len(X)
        A = X.conj().T @ X + lam * np.eye(X.shape[1])
        coef = np.linalg.solve(A, X.conj().T @ y)
        self.hybrid = coef
        C.ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        HYBRID_FILE.write_text(json.dumps({"re": coef.real.tolist(), "im": coef.imag.tolist(),
                                           "features": ["wind", "|wind|*wind/10", "current", "small_berg*wind"],
                                           "n_samples": int(len(X))}))
        log(f"hybrid residual model fitted on {len(X)} 6-hour windows; coef={np.round(coef, 4)}")
        return self.evaluate(log=log)

    def evaluate(self, n_days: int = 24, per_day: int = 10, log=print) -> dict:
        """72 h end-point error on the held-out 2025 year using forecast winds."""
        rng = np.random.default_rng(2)
        va_end = C.day_index(C.VAL_END)
        days = list(rng.choice(np.arange(va_end + 5, self.s.n_days - 4), n_days, replace=False))
        e_p, e_h, e_pers = [], [], []
        for day in days:
            b = self.seed_bergs(day, per_day)
            t0 = day * 24.0
            sic = self.sic_sampler(day, [self.s.sic_truth(min(day + k, self.s.n_days - 1)) for k in range(4)])
            tr, _ = self.simulate(b, t0, 72, truth=True, sic_fn=sic, record_every_h=72)
            ph, _ = self.simulate(b, t0, 72, truth=False, issue_h=t0, sic_fn=sic, record_every_h=72)
            hy, _ = self.simulate(b, t0, 72, truth=False, issue_h=t0, sic_fn=sic, hybrid=True, record_every_h=72)
            e_p += list(np.hypot(*(ph[-1] - tr[-1]).T))
            e_h += list(np.hypot(*(hy[-1] - tr[-1]).T))
            e_pers += list(np.hypot(*(tr[0] - tr[-1]).T))
        m = {"n_tracks": len(e_p), "horizon_h": 72,
             "persistence_km": {"mean": float(np.mean(e_pers)), "median": float(np.median(e_pers))},
             "physics_km": {"mean": float(np.mean(e_p)), "median": float(np.median(e_p))},
             "hybrid_km": {"mean": float(np.mean(e_h)), "median": float(np.median(e_h))}}
        METRICS_FILE.write_text(json.dumps(m, indent=2))
        log(f"72 h drift error on 2025 (km, mean/median): persistence {m['persistence_km']['mean']:.1f}/"
            f"{m['persistence_km']['median']:.1f}  physics {m['physics_km']['mean']:.1f}/{m['physics_km']['median']:.1f}"
            f"  hybrid {m['hybrid_km']['mean']:.1f}/{m['hybrid_km']['median']:.1f}")
        return m

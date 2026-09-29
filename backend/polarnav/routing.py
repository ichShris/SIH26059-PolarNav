"""Time-dependent, fuel-optimal A* routing on the 50 km navigation grid.

Edge cost (tonnes-of-fuel equivalent):
    cost = fuel(edge) + w_time * hours(edge) + w_risk * km * max(0, RIO_BUFFER - RIO) / 10
           + w_ice * km * SIC + iceberg penalty
Risk aversion penalises sailing below a RIO safety buffer (10), not only below the POLARIS
limit of 0, so a more risk-averse master trades distance for ice with more margin.
where fuel and speed come from fuel.py using the sea-ice field *valid at the ship's ETA*
(analysis for the first 24 h, then the ConvLSTM +24/+48/+72 h forecasts; the +72 h
field is held beyond the forecast horizon). Cells with RIO in "special consideration"
or where the ship would be beset are blocked. Iceberg exclusion zones move with the
ensemble-mean drift forecast and grow with its 2-sigma uncertainty.

The baseline is what a navigator does with a static ice chart: the shortest path that
avoids land and today's special-consideration ice, with no fuel model and no iceberg
forecast. Both routes are then evaluated with the same ship model and forecasts.
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from . import config as C
from . import fuel, polaris
from .projection import bearing_deg, scale_factor, to_latlon, to_xy

NB = [(di, dj) for di in (-2, -1, 0, 1, 2) for dj in (-2, -1, 0, 1, 2)
      if (di, dj) != (0, 0) and math.gcd(abs(di), abs(dj)) == 1]  # 16-connected
RIO_BUFFER = 10.0         # RIO below this is penalised in proportion to w_risk
BERG_SOFT_KM = 45.0       # penalty zone beyond the hard exclusion radius
BERG_PENALTY_T = 3.0      # t-fuel-equivalent at the edge of the hard zone


@dataclass
class RouteRequest:
    day: int
    origin: tuple[float, float]
    dest: tuple[float, float]
    ice_class: str = "PC5"
    cruise_kn: float = 12.0
    w_time: float = 0.4      # t fuel equivalent per hour (charter + science-day value)
    w_risk: float = 0.02     # t per km per 10 RIO points below RIO_BUFFER
    avoid_bergs: bool = True
    w_ice: float = 0.0       # t per km per unit SIC (ice-exposure aversion)
    berg_margin_nm: float = 5.0  # clearance beyond berg size + 2-sigma drift uncertainty


class Environment:
    """Per-time-slice cell costs for one request."""

    def __init__(self, store, sic_slices: list[np.ndarray], day: int, req: RouteRequest):
        self.store = store
        blocked = store.masks["blocked"]
        self.blocked = blocked
        self.n = C.NAV_N
        gx, gy = np.meshgrid(C.NAV_X, C.NAV_Y)
        self.k = scale_factor(to_latlon(gx, gy)[0])
        self.slices = []
        for s, ice in enumerate(sic_slices):
            thick = store.thickness(ice, day + s)
            sic_n, thick_n = store.embed(ice), store.embed(thick)
            r = polaris.rio(sic_n, thick_n, req.ice_class)
            lvl = polaris.operation_level(r, req.ice_class)
            v, fpk, hpk, beset = fuel.cell_performance(sic_n, thick_n, req.ice_class, req.cruise_kn,
                                                       lvl == polaris.ELEVATED)
            hard = blocked | (lvl == polaris.SPECIAL) | beset
            self.slices.append({"sic": sic_n, "thick": thick_n, "rio": r, "level": lvl, "speed": v,
                                "fuel_pk": fpk, "hours_pk": hpk, "hard": hard})

    def snap(self, lat: float, lon: float) -> tuple[int, int]:
        x, y = to_xy(lat, lon)
        i = int(np.clip((x + C.NAV_HALF_KM) // C.CELL_KM, 0, self.n - 1))
        j = int(np.clip((y + C.NAV_HALF_KM) // C.CELL_KM, 0, self.n - 1))
        if not self.blocked[j, i]:
            return j, i
        _, (jj, ii) = ndimage.distance_transform_edt(self.blocked, return_indices=True)
        return int(jj[j, i]), int(ii[j, i])


class BergField:
    """Moving iceberg exclusion zones from the drift forecast."""

    def __init__(self, bergs: list[dict], step_h: float, margin_km: float = 9.26):
        self.step_h = step_h
        self.margin_km = margin_km
        self.tracks = [np.asarray(b["forecast"]) for b in bergs]
        self.sigma = []
        self.radius = []
        self.ids = [b["id"] for b in bergs]
        for b in bergs:
            ell = b["ellipses"]
            t = np.array([e["t_h"] for e in ell], float)
            a = np.array([e["a"] for e in ell], float)
            self.sigma.append((t, a))
            self.radius.append(b["length_m"] / 2000.0)
        # spatial index: cell -> berg ids whose swept path passes within reach
        self.index: dict[int, list[int]] = {}
        reach = BERG_SOFT_KM + 60.0
        for bi, tr in enumerate(self.tracks):
            cells = set()
            for x, y in tr:
                i0 = int((x - reach + C.NAV_HALF_KM) // C.CELL_KM)
                i1 = int((x + reach + C.NAV_HALF_KM) // C.CELL_KM)
                j0 = int((y - reach + C.NAV_HALF_KM) // C.CELL_KM)
                j1 = int((y + reach + C.NAV_HALF_KM) // C.CELL_KM)
                for j in range(max(j0, 0), min(j1, C.NAV_N - 1) + 1):
                    for i in range(max(i0, 0), min(i1, C.NAV_N - 1) + 1):
                        cells.add(j * C.NAV_N + i)
            for c in cells:
                self.index.setdefault(c, []).append(bi)

    def position(self, bi: int, t_h: float):
        tr = self.tracks[bi]
        f = min(max(t_h / self.step_h, 0.0), len(tr) - 1.0)
        k = int(f)
        if k >= len(tr) - 1:
            return tr[-1]
        w = f - k
        return tr[k] * (1 - w) + tr[k + 1] * w

    def exclusion_km(self, bi: int, t_h: float) -> float:
        t, a = self.sigma[bi]
        return self.radius[bi] + float(np.interp(t_h, t, a)) + self.margin_km

    def penalty(self, cell: int, x: float, y: float, t_h: float) -> float:
        ids = self.index.get(cell)
        if not ids:
            return 0.0
        pen = 0.0
        for bi in ids:
            px, py = self.position(bi, t_h)
            d = math.hypot(x - px, y - py)
            r = self.exclusion_km(bi, t_h)
            if d < r:
                return math.inf
            if d < r + BERG_SOFT_KM:
                pen += BERG_PENALTY_T * (1.0 - (d - r) / BERG_SOFT_KM)
        return pen


def astar(env: Environment, start, goal, req: RouteRequest, bergs: BergField | None, mode: str):
    """mode 'optimal' = fuel/time/risk/bergs; 'shortest' = distance with day-0 hard limits only."""
    n = env.n
    sl = env.slices
    ns = len(sl)
    kf = env.k.ravel().tolist()
    hard = [s["hard"].ravel().tolist() for s in sl]
    blocked = env.blocked.ravel().tolist()
    fuel_pk = [s["fuel_pk"].ravel().tolist() for s in sl]
    hours_pk = [s["hours_pk"].ravel().tolist() for s in sl]
    rio = [s["rio"].ravel().tolist() for s in sl]
    sic = [s["sic"].ravel().tolist() for s in sl]
    xs, ys = C.NAV_X.tolist(), C.NAV_Y.tolist()
    s_idx = start[0] * n + start[1]
    g_idx = goal[0] * n + goal[1]
    gx, gy = xs[goal[1]], ys[goal[0]]

    if mode == "optimal":
        per_km = min(float(np.min((s["fuel_pk"] + req.w_time * s["hours_pk"])[~s["hard"]])) for s in sl)
    else:
        per_km = 1.0

    def h(j, i):
        dx, dy = xs[i] - gx, ys[j] - gy
        return math.hypot(dx, dy) / max(kf[j * n + i], kf[g_idx]) * per_km

    g = {s_idx: 0.0}
    t_arr = {s_idx: 0.0}
    parent = {s_idx: -1}
    openh = [(h(*start), 0.0, s_idx)]
    closed = set()
    while openh:
        f, gc, u = heapq.heappop(openh)
        if u in closed:
            continue
        if u == g_idx:
            break
        closed.add(u)
        uj, ui = divmod(u, n)
        tu = t_arr[u]
        slot = min(int(tu // 24.0), ns - 1)
        for dj, di in NB:
            vj, vi = uj + dj, ui + di
            if not (0 <= vj < n and 0 <= vi < n):
                continue
            v = vj * n + vi
            if v in closed:
                continue
            if mode == "optimal":
                hard_v = hard[slot][v]
            else:
                hard_v = hard[0][v]
            if (hard_v or blocked[v]) and v != g_idx:
                continue
            # no corner cutting across land for long/diagonal moves
            if abs(dj) + abs(di) > 1:
                sj, si = (1 if dj > 0 else -1), (1 if di > 0 else -1)
                if abs(dj) == 2:
                    mids = ((uj + sj) * n + ui, (uj + sj) * n + vi)
                elif abs(di) == 2:
                    mids = (uj * n + ui + si, vj * n + ui + si)
                else:
                    mids = ((uj + sj) * n + ui, uj * n + ui + si)
                if blocked[mids[0]] or blocked[mids[1]]:
                    continue
            km = math.hypot(xs[vi] - xs[ui], ys[vj] - ys[uj]) * 2.0 / (kf[u] + kf[v])
            if mode == "optimal":
                hrs = km * 0.5 * (hours_pk[slot][u] + hours_pk[slot][v])
                cost = km * 0.5 * (fuel_pk[slot][u] + fuel_pk[slot][v]) + req.w_time * hrs
                r = rio[slot][v]
                if r < RIO_BUFFER:
                    cost += req.w_risk * km * (RIO_BUFFER - r) / 10.0
                if req.w_ice:
                    cost += req.w_ice * km * sic[slot][v]
                if bergs is not None and req.avoid_bergs:
                    p = bergs.penalty(v, xs[vi], ys[vj], tu + hrs)
                    if p == math.inf and v != g_idx:
                        continue
                    cost += p
            else:
                hrs = km / (req.cruise_kn * 1.852)
                cost = km
            ng = gc + cost
            if ng < g.get(v, math.inf):
                g[v] = ng
                t_arr[v] = tu + hrs
                parent[v] = u
                heapq.heappush(openh, (ng + h(vj, vi), ng, v))
    if g_idx not in parent:
        return None, len(closed)
    path = []
    c = g_idx
    while c != -1:
        path.append(divmod(c, n))
        c = parent[c]
    return path[::-1], len(closed)


def evaluate(env: Environment, path, req: RouteRequest, berg_scn: list[dict] | None, step_h: float):
    """Walk the path with the ship model to get ETAs, fuel, risk and iceberg clearances."""
    n = env.n
    t, fuel_t, dist = 0.0, 0.0, 0.0
    pts = []
    worst_level = 0
    min_rio = 99.0
    elevated_km = 0.0
    beyond_horizon = False
    for idx, (j, i) in enumerate(path):
        x, y = float(C.NAV_X[i]), float(C.NAV_Y[j])
        slot = min(int(t // 24.0), len(env.slices) - 1)
        if t >= 24.0 * len(env.slices):
            beyond_horizon = True
        s = env.slices[slot]
        if idx > 0:
            pj, pi = path[idx - 1]
            km = math.hypot(x - C.NAV_X[pi], y - C.NAV_Y[pj]) * 2.0 / (env.k[j, i] + env.k[pj, pi])
            hrs = km * 0.5 * (s["hours_pk"][j, i] + s["hours_pk"][pj, pi])
            fuel_t += km * 0.5 * (s["fuel_pk"][j, i] + s["fuel_pk"][pj, pi])
            t += hrs
            dist += km
            if s["level"][j, i] == polaris.ELEVATED:
                elevated_km += km
        lat, lon = to_latlon(x, y)
        lvl = int(s["level"][j, i])
        worst_level = max(worst_level, lvl)
        min_rio = min(min_rio, float(s["rio"][j, i]))
        pts.append({"x": x, "y": y, "lat": round(float(lat), 3), "lon": round(float(lon), 3),
                    "t_h": round(t, 2), "sic": round(float(s["sic"][j, i]), 3),
                    "thick_m": round(float(s["thick"][j, i]), 2), "rio": round(float(s["rio"][j, i]), 1),
                    "level": lvl, "speed_kn": round(float(s["speed"][j, i]), 1),
                    "fuel_t": round(fuel_t, 2), "dist_nm": round(dist / 1.852, 1)})

    clear = []
    if berg_scn:
        tt = np.array([p["t_h"] for p in pts])
        px = np.array([p["x"] for p in pts])
        py = np.array([p["y"] for p in pts])
        # densify in time so fast legs are not skipped
        tq = np.arange(0, tt[-1] + 0.5, 0.5)
        qx, qy = np.interp(tq, tt, px), np.interp(tq, tt, py)
        for b in berg_scn:
            for key in ("forecast", "truth"):
                tr = np.asarray(b[key])
                f = np.clip(tq / step_h, 0, len(tr) - 1)
                k0 = np.floor(f).astype(int)
                k1 = np.minimum(k0 + 1, len(tr) - 1)
                w = (f - k0)[:, None]
                bp = tr[k0] * (1 - w) + tr[k1] * w
                d = np.hypot(qx - bp[:, 0], qy - bp[:, 1]) - b["length_m"] / 2000.0
                m = int(np.argmin(d))
                if key == "forecast":
                    rec = {"id": b["id"], "cpa_km": round(float(d[m]), 1), "t_h": round(float(tq[m]), 1)}
                else:
                    rec["cpa_truth_km"] = round(float(d[m]), 1)
            clear.append(rec)
        clear.sort(key=lambda r: r["cpa_km"])

    wps = key_waypoints(pts)
    return {
        "points": pts,
        "waypoints": wps,
        "summary": {
            "distance_nm": round(dist / 1.852, 1),
            "hours": round(t, 1),
            "days": round(t / 24.0, 2),
            "fuel_t": round(fuel_t, 1),
            "co2_t": round(fuel_t * C.VESSEL.co2_per_t_fuel, 1),
            "min_rio": round(min_rio, 1),
            "worst_level": polaris.LEVEL_NAMES[worst_level],
            "elevated_nm": round(elevated_km / 1.852, 1),
            "ice_nm": round(sum(1 for p in pts if p["sic"] > 0.15) * C.CELL_KM / 1.852, 0),
            "beyond_forecast_horizon": beyond_horizon,
            "min_berg_cpa_km": clear[0]["cpa_km"] if clear else None,
            "min_berg_cpa_truth_km": min((c["cpa_truth_km"] for c in clear), default=None),
            "bergs_within_10nm": sum(1 for c in clear if c["cpa_km"] < 18.52),
        },
        "berg_clearance": clear[:8],
    }


def key_waypoints(pts, tol_km: float = 25.0):
    """Douglas-Peucker thinning of the cell path into navigational waypoints."""
    xy = np.array([[p["x"], p["y"]] for p in pts])

    def dp(a, b, keep):
        if b <= a + 1:
            return
        seg = xy[b] - xy[a]
        L = np.hypot(*seg) + 1e-9
        rel = xy[a + 1:b] - xy[a]
        d = np.abs(seg[0] * rel[:, 1] - seg[1] * rel[:, 0]) / L
        k = int(np.argmax(d))
        if d[k] > tol_km:
            m = a + 1 + k
            keep.add(m)
            dp(a, m, keep)
            dp(m, b, keep)

    keep = {0, len(pts) - 1}
    dp(0, len(pts) - 1, keep)
    idx = sorted(keep)
    out = []
    for n_, k in enumerate(idx):
        p = pts[k]
        crs = None
        if n_ + 1 < len(idx):
            q = pts[idx[n_ + 1]]
            crs = round(float(bearing_deg(p["x"], p["y"], q["x"], q["y"])), 0)
        out.append({"name": f"P{n_ + 1}", **{k2: p[k2] for k2 in ("lat", "lon", "x", "y", "t_h", "sic", "rio", "speed_kn", "dist_nm", "fuel_t")},
                    "course": crs})
    return out


def alternative_profiles(req: RouteRequest):
    """Route options computed alongside the user's fuel-optimal route."""
    from dataclasses import replace
    return [
        ("fastest", "Fastest", replace(req, w_time=max(req.w_time, 4.0))),
        ("safest", "Least ice exposure", replace(req, w_risk=max(req.w_risk * 5, 0.25), w_ice=max(req.w_ice, 0.3),
                                                berg_margin_nm=max(req.berg_margin_nm, 12.0))),
    ]


def plan(store, sic_slices, req: RouteRequest, berg_scn: dict | None) -> dict:
    env = Environment(store, sic_slices, req.day, req)
    start = env.snap(*req.origin)
    goal = env.snap(*req.dest)
    bergs = BergField(berg_scn["bergs"], berg_scn["step_h"], req.berg_margin_nm * 1.852) if berg_scn else None
    warnings = []
    opt_path, n_opt = astar(env, start, goal, req, bergs, "optimal")
    if opt_path is None:
        warnings.append("No route satisfies POLARIS limits and iceberg clearances for this ice class; "
                        "consider a higher ice class, icebreaker escort, or a later departure.")
    base_path, n_base = astar(env, start, goal, req, None, "shortest")
    step = berg_scn["step_h"] if berg_scn else 3
    out = {"request": {**req.__dict__}, "warnings": warnings,
           "snapped": {"origin": [float(C.NAV_X[start[1]]), float(C.NAV_Y[start[0]])],
                       "dest": [float(C.NAV_X[goal[1]]), float(C.NAV_Y[goal[0]])]},
           "expanded": {"optimal": n_opt, "baseline": n_base}}
    berg_list = berg_scn["bergs"] if berg_scn else None
    out["optimal"] = evaluate(env, opt_path, req, berg_list, step) if opt_path else None
    out["baseline"] = evaluate(env, base_path, req, berg_list, step) if base_path else None
    out["alternatives"] = []
    for key, label, alt in alternative_profiles(req):
        ab = BergField(berg_scn["bergs"], berg_scn["step_h"], alt.berg_margin_nm * 1.852) if berg_scn else None
        path, _ = astar(env, start, goal, alt, ab, "optimal")
        if path:
            ev = evaluate(env, path, alt, berg_list, step)
            ev.update({"key": key, "label": label,
                       "same_track_as_optimal": bool(opt_path) and path == opt_path})
            out["alternatives"].append(ev)
    if out["optimal"] and out["baseline"]:
        o, b = out["optimal"]["summary"], out["baseline"]["summary"]
        out["comparison"] = {
            "fuel_saving_t": round(b["fuel_t"] - o["fuel_t"], 1),
            "fuel_saving_pct": round(100.0 * (b["fuel_t"] - o["fuel_t"]) / max(b["fuel_t"], 1e-6), 1),
            "co2_saving_t": round(b["co2_t"] - o["co2_t"], 1),
            "time_delta_h": round(o["hours"] - b["hours"], 1),
            "distance_delta_nm": round(o["distance_nm"] - b["distance_nm"], 1),
        }
    return out

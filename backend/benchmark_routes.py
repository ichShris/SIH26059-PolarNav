"""Season benchmark: PolarNav route vs conventional shortest route over the held-out 2025 seasons.

For every 4th day of the Nov-Feb sailing windows in the test year, plan the ice passage
from an ice-edge approach point to Maitri and to Bharati, and record fuel, time and
iceberg clearance for both routes. Writes artifacts/route_benchmark.json.

    python benchmark_routes.py [--ice-class PC5] [--step 4]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json

import numpy as np

from polarnav import config as C
from polarnav.api import berg_scenario, sic_bundle, svc
from polarnav.routing import RouteRequest, plan

LEGS = {
    "maitri": ((-58.0, 14.0), (C.PLACES["maitri"]["lat"], C.PLACES["maitri"]["lon"])),
    "bharati": ((-58.0, 78.0), (C.PLACES["bharati"]["lat"], C.PLACES["bharati"]["lon"])),
}
OUT = C.ARTIFACT_DIR / "route_benchmark.json"


def season_days(step: int) -> list[int]:
    days = []
    for a, b in ((dt.date(2025, 1, 1), dt.date(2025, 2, 15)), (dt.date(2025, 11, 1), dt.date(2025, 12, 28))):
        d = a
        while d <= b:
            days.append(C.day_index(d))
            d += dt.timedelta(days=step)
    return days


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ice-class", default="PC5")
    ap.add_argument("--step", type=int, default=4)
    args = ap.parse_args()
    store = svc()["store"]
    rows = []
    for day in season_days(args.step):
        for leg, (o, d) in LEGS.items():
            req = RouteRequest(day=day, origin=o, dest=d, ice_class=args.ice_class)
            p = plan(store, sic_bundle(day), req, berg_scenario(day))
            if not (p["optimal"] and p["baseline"]):
                rows.append({"date": str(C.index_day(day)), "leg": leg, "feasible": False})
                continue
            o_s, b_s = p["optimal"]["summary"], p["baseline"]["summary"]
            rows.append({"date": str(C.index_day(day)), "leg": leg, "feasible": True,
                         "fuel_saving_pct": p["comparison"]["fuel_saving_pct"],
                         "time_delta_h": p["comparison"]["time_delta_h"],
                         "opt_fuel_t": o_s["fuel_t"], "base_fuel_t": b_s["fuel_t"],
                         "ice_nm": b_s["ice_nm"],
                         "opt_min_cpa_truth_km": o_s["min_berg_cpa_truth_km"],
                         "base_min_cpa_truth_km": b_s["min_berg_cpa_truth_km"]})
            r = rows[-1]
            print(f"{r['date']} {leg:8s} saving {r['fuel_saving_pct']:5.1f}%  dt {r['time_delta_h']:+6.1f} h  "
                  f"ice {r['ice_nm']:4.0f} nm  CPA truth opt {r['opt_min_cpa_truth_km']} base {r['base_min_cpa_truth_km']}", flush=True)
    ok = [r for r in rows if r["feasible"]]
    s = np.array([r["fuel_saving_pct"] for r in ok])
    icy = np.array([r["fuel_saving_pct"] for r in ok if r["ice_nm"] >= 150])
    close = lambda k: sum(1 for r in ok if r[k] is not None and r[k] < 18.52)
    summary = {
        "ice_class": args.ice_class, "voyages": len(rows), "feasible": len(ok),
        "fuel_saving_pct": {"mean": float(s.mean()), "median": float(np.median(s)),
                            "p10": float(np.percentile(s, 10)), "p90": float(np.percentile(s, 90)),
                            "max": float(s.max())},
        "fuel_saving_pct_heavy_ice": {"n": int(len(icy)), "mean": float(icy.mean()) if len(icy) else None,
                                      "median": float(np.median(icy)) if len(icy) else None},
        "mean_time_saved_h": float(-np.mean([r["time_delta_h"] for r in ok])),
        "voyages_passing_berg_within_10nm": {"polarnav": close("opt_min_cpa_truth_km"),
                                             "conventional": close("base_min_cpa_truth_km")},
    }
    OUT.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

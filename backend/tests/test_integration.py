"""Integration tests against the built dataset and trained models (skipped if not built)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from polarnav import config as C  # noqa: E402
from polarnav.data import WORLD_FILE  # noqa: E402
from polarnav.sic_model import MODEL_FILE  # noqa: E402

pytestmark = pytest.mark.skipif(not (WORLD_FILE.exists() and MODEL_FILE.exists()),
                                reason="run backend/train.py first")


@pytest.fixture(scope="module")
def plan():
    from polarnav.api import berg_scenario, ctx, sic_bundle
    from polarnav.routing import RouteRequest
    from polarnav.routing import plan as make_plan
    day = C.day_index(C.DEFAULT_DATE)
    req = RouteRequest(day=day, origin=(-58.0, 14.0), dest=(-69.9, 11.75), ice_class="PC5")
    c = ctx("sim")
    return make_plan(c.store, sic_bundle(c.key, day), req, berg_scenario(c.key, day)), berg_scenario(c.key, day), c.store


def test_routes_stay_off_land(plan):
    p, _, store = plan
    blocked = store.masks["blocked"]
    for key in ("optimal", "baseline"):
        pts = p[key]["points"]
        for q in pts[1:-1]:  # endpoints are snapped to the nearest navigable cell
            i = int((q["x"] + C.NAV_HALF_KM) // C.CELL_KM)
            j = int((q["y"] + C.NAV_HALF_KM) // C.CELL_KM)
            assert not blocked[j, i]


def test_baseline_is_shortest_and_polarnav_respects_limits(plan):
    p, _, _ = plan
    o, b = p["optimal"]["summary"], p["baseline"]["summary"]
    assert b["distance_nm"] <= o["distance_nm"] + 1.0
    assert o["worst_level"] != "Operation subject to special consideration"


def test_forecast_iceberg_clearance(plan):
    p, bergs, _ = plan
    # every forecast CPA must clear the berg's hard exclusion radius (5 nm margin + size + uncertainty)
    for c in p["optimal"]["berg_clearance"]:
        assert c["cpa_km"] > 9.0


def test_drift_ensemble_shapes(plan):
    _, bergs, _ = plan
    b = bergs["bergs"][0]
    assert len(b["forecast"]) == 72 // bergs["step_h"] + 1
    assert len(b["members_72h"]) == bergs["members"]
    assert np.isfinite(np.asarray(b["forecast"])).all()

"""Vessel speed / power / fuel model in open water and pack ice.

Open water:  P_ow(v) = P_max (v / V_max)^3                       (propulsion ~ cube law)
Pack ice:    P_ice(v) = beta (h_eff / h_cap)^1.5 (1 + gamma v) v  (breaking + submersion + friction)
with the equivalent thickness h_eff = h * ((C - 0.1) / 0.9)^1.5 so that open drift ice
below ~10% concentration is navigated around rather than broken. beta is calibrated so
the ship makes ~3 kn at full power in level ice equal to its rated capability h_cap —
the usual way ice classes are specified. Speed is the lesser of the requested cruise
speed, the attainable speed at full power, and the POLARIS limit for elevated risk.
"""
from __future__ import annotations

import numpy as np

from .config import ICE_CLASSES, KNOT_MS, VESSEL

GAMMA = 0.5          # 1 / (m/s)
MIN_SPEED_KN = 1.0   # below this the ship is considered beset


def _beta() -> float:
    v = 3.0 * KNOT_MS
    p_ow = VESSEL.max_power_kw * (3.0 / VESSEL.max_speed_kn) ** 3
    return (VESSEL.max_power_kw - p_ow) * 1000.0 / ((1 + GAMMA * v) * v)


BETA = _beta()


def equivalent_thickness(sic, thickness_m):
    c = np.clip((np.asarray(sic) - 0.1) / 0.9, 0.0, 1.0)
    return np.asarray(thickness_m) * c ** 1.5


def power_kw(v_kn, h_eff, cap_m):
    v = np.asarray(v_kn) * KNOT_MS
    p_ow = VESSEL.max_power_kw * (np.asarray(v_kn) / VESSEL.max_speed_kn) ** 3
    p_ice = BETA * (np.asarray(h_eff) / cap_m) ** 1.5 * (1 + GAMMA * v) * v / 1000.0
    return p_ow + p_ice


def attainable_speed_kn(h_eff, cap_m):
    """Speed at full power in ice of equivalent thickness h_eff (vectorised bisection)."""
    h_eff = np.asarray(h_eff, dtype=float)
    lo = np.zeros_like(h_eff)
    hi = np.full_like(h_eff, VESSEL.max_speed_kn)
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        over = power_kw(mid, h_eff, cap_m) > VESSEL.max_power_kw
        hi = np.where(over, mid, hi)
        lo = np.where(over, lo, mid)
    return lo


def cell_performance(sic, thickness_m, ice_class: str, cruise_kn: float, elevated_mask):
    """Per-cell speed (kn), fuel per true km (t/km) and beset flag."""
    cls = ICE_CLASSES[ice_class]
    h_eff = equivalent_thickness(sic, thickness_m)
    v = np.minimum(cruise_kn, attainable_speed_kn(h_eff, cls.capability_m))
    v = np.where(elevated_mask, np.minimum(v, cls.elevated_speed_kn), v)
    beset = v < MIN_SPEED_KN
    v = np.maximum(v, MIN_SPEED_KN)
    p = power_kw(v, h_eff, cls.capability_m) + VESSEL.hotel_kw
    fuel_t_per_h = p * VESSEL.sfoc_g_per_kwh / 1e6
    hours_per_km = 1.0 / (v * 1.852)
    return v, fuel_t_per_h * hours_per_km, hours_per_km, beset

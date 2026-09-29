"""Static configuration: grids, time span, vessel classes and named locations."""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "backend" / "data"
ARTIFACT_DIR = ROOT / "backend" / "artifacts"
FRONTEND_DIST = ROOT / "frontend" / "dist"

# --------------------------------------------------------------------------- grids
# Navigation grid: covers the Southern Ocean out to Cape Town / Mauritius.
CELL_KM = 50.0
NAV_HALF_KM = 7500.0
NAV_N = int(2 * NAV_HALF_KM / CELL_KM)  # 300
NAV_X = -NAV_HALF_KM + CELL_KM * (np.arange(NAV_N) + 0.5)  # cell centres, west -> east
NAV_Y = -NAV_HALF_KM + CELL_KM * (np.arange(NAV_N) + 0.5)  # cell centres, south -> north (row 0 = min y)

# Sea-ice model grid: the inner crop of the navigation grid where ice can exist
# (reaches ~54 S along the axes, beyond the observed maximum winter ice edge).
ICE_HALF_KM = 4000.0
ICE_N = int(2 * ICE_HALF_KM / CELL_KM)  # 160
ICE_OFF = (NAV_N - ICE_N) // 2  # index offset of the ice grid inside the nav grid

# --------------------------------------------------------------------------- time
SIM_START = dt.date(2022, 1, 1)
SIM_END = dt.date(2025, 12, 31)
TRAIN_END = dt.date(2024, 6, 30)   # ConvLSTM trains on dates up to here
VAL_END = dt.date(2024, 12, 31)    # validation window; 2025 is a held-out test year
DEFAULT_DATE = dt.date(2025, 11, 20)  # early Indian Antarctic Expedition season, held-out test year

HISTORY_DAYS = 7
FORECAST_DAYS = 3  # 72 h


def day_index(d: dt.date) -> int:
    return (d - SIM_START).days


def index_day(i: int) -> dt.date:
    return SIM_START + dt.timedelta(days=int(i))


N_DAYS = day_index(SIM_END) + 1

# --------------------------------------------------------------------------- vessel
KNOT_MS = 0.514444
NM_KM = 1.852


@dataclass(frozen=True)
class IceClass:
    name: str
    polar_class: bool          # IACS Polar Class (PC1-PC7) vs Baltic / no ice class
    capability_m: float        # level-ice thickness it breaks continuously at ~3 kn
    elevated_speed_kn: float   # POLARIS recommended max speed in elevated-risk ice


# Capabilities are indicative values for a ~10 MW research / supply vessel.
ICE_CLASSES: dict[str, IceClass] = {c.name: c for c in [
    IceClass("PC1", True, 3.0, 11.0),
    IceClass("PC2", True, 2.5, 8.0),
    IceClass("PC3", True, 2.0, 5.0),
    IceClass("PC4", True, 1.5, 5.0),
    IceClass("PC5", True, 1.0, 5.0),
    IceClass("PC6", True, 0.8, 3.0),
    IceClass("PC7", True, 0.6, 3.0),
    IceClass("IA Super", False, 1.0, 3.0),
    IceClass("IA", False, 0.8, 3.0),
    IceClass("IB", False, 0.6, 3.0),
    IceClass("IC", False, 0.4, 3.0),
    IceClass("No Ice Class", False, 0.2, 3.0),
]}


@dataclass(frozen=True)
class Vessel:
    name: str = "RV Sagar Hima (notional PRV)"
    max_power_kw: float = 10_000.0
    max_speed_kn: float = 16.0
    sfoc_g_per_kwh: float = 205.0     # specific fuel oil consumption, MGO
    co2_per_t_fuel: float = 3.206     # IMO MEPC.308(73) factor for MGO
    hotel_kw: float = 600.0           # auxiliary load, burns fuel regardless of speed


VESSEL = Vessel()

# --------------------------------------------------------------------------- places
PLACES = {
    "vessel": {"name": "Vessel (en route, 58S 14E)", "lat": -58.0, "lon": 14.0, "kind": "vessel"},
    "cape_town": {"name": "Cape Town", "lat": -33.90, "lon": 18.40, "kind": "port"},
    "port_louis": {"name": "Port Louis, Mauritius", "lat": -20.15, "lon": 57.48, "kind": "port"},
    "hobart": {"name": "Hobart", "lat": -42.88, "lon": 147.34, "kind": "port"},
    "maitri": {"name": "Maitri (India Bay offload)", "lat": -69.90, "lon": 11.75, "kind": "station"},
    "bharati": {"name": "Bharati (Larsemann Hills)", "lat": -69.41, "lon": 76.19, "kind": "station"},
}

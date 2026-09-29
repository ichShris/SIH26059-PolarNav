"""POLARIS — Polar Operational Limit Assessment Risk Indexing System (IMO MSC.1/Circ.1519).

RIO = sum_i C_i * RIV_i, with C_i the concentration of ice type i in tenths and RIV_i the
Risk Index Value for the ship's ice class. Operation levels:
    Polar Class ships:   RIO >= 0 normal | -10 <= RIO < 0 elevated risk | RIO < -10 special consideration
    other ice classes:   RIO >= 0 normal | RIO < 0 special consideration

The RIV table below is transcribed from MSC.1/Circ.1519 Table 1.3; verify against the
circular before operational use.
"""
from __future__ import annotations

import numpy as np

from .config import ICE_CLASSES

ICE_TYPES = [
    "Ice free", "New ice", "Grey ice", "Grey-white ice", "Thin FY 1st stage", "Thin FY 2nd stage",
    "Medium FY <1 m", "Medium FY", "Thick FY", "Second-year", "Light multi-year", "Heavy multi-year",
]
# upper thickness bound (m) of each ice type, used to classify modelled thickness
THICKNESS_EDGES = np.array([0.0, 0.10, 0.15, 0.30, 0.50, 0.70, 1.00, 1.20, 2.00, 2.50, 3.20])

RIV = {
    "PC1": [3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 1, 1],
    "PC2": [3, 3, 3, 3, 2, 2, 2, 2, 2, 1, 1, 0],
    "PC3": [3, 3, 3, 3, 2, 2, 2, 2, 2, 1, 0, -1],
    "PC4": [3, 3, 3, 3, 2, 2, 2, 2, 1, 0, -1, -2],
    "PC5": [3, 3, 3, 3, 2, 2, 1, 1, 0, -1, -2, -2],
    "PC6": [3, 2, 2, 2, 2, 1, 1, 0, -1, -2, -3, -3],
    "PC7": [3, 2, 2, 2, 1, 1, 0, -1, -2, -3, -3, -3],
    "IA Super": [3, 2, 2, 2, 2, 1, 1, 0, -1, -2, -3, -3],
    "IA": [3, 2, 2, 2, 1, 1, 0, -1, -2, -3, -4, -4],
    "IB": [3, 2, 2, 1, 1, 0, -1, -2, -3, -4, -5, -5],
    "IC": [3, 2, 1, 0, 0, -1, -2, -3, -4, -5, -6, -6],
    "No Ice Class": [3, 1, 0, -1, -1, -2, -3, -4, -5, -6, -7, -8],
}

NORMAL, ELEVATED, SPECIAL = 0, 1, 2
LEVEL_NAMES = {NORMAL: "Normal operation", ELEVATED: "Elevated risk", SPECIAL: "Operation subject to special consideration"}


def ice_type(thickness_m: np.ndarray) -> np.ndarray:
    """Stage-of-development index (1..11) from thickness; 0 where there is no ice."""
    t = np.searchsorted(THICKNESS_EDGES, np.asarray(thickness_m), side="right")
    return np.clip(t, 1, 11)


def rio(sic: np.ndarray, thickness_m: np.ndarray, ice_class: str) -> np.ndarray:
    """Risk Index Outcome per cell.

    Each cell's ice is split 70/30 between its modelled stage of development and the
    next thinner stage, a simple stand-in for the partial concentrations on an ice chart.
    """
    riv = np.asarray(RIV[ice_class], dtype=float)
    c = np.round(np.clip(sic, 0, 1) * 10.0)  # tenths, as reported on ice charts
    t = ice_type(thickness_m)
    thinner = np.maximum(t - 1, 1)
    ice_riv = 0.7 * riv[t] + 0.3 * riv[thinner]
    return (10.0 - c) * riv[0] + c * ice_riv


def operation_level(r: np.ndarray, ice_class: str) -> np.ndarray:
    r = np.asarray(r)
    if ICE_CLASSES[ice_class].polar_class:
        return np.where(r >= 0, NORMAL, np.where(r >= -10, ELEVATED, SPECIAL))
    return np.where(r >= 0, NORMAL, SPECIAL)

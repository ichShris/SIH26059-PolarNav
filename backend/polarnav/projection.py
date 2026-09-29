"""South polar stereographic projection (EPSG:3031 on a sphere, true scale at 71 deg S).

All internal geometry runs in projected kilometres. The spherical form differs from
the ellipsoidal EPSG:3031 by well under one grid cell at 50 km resolution, which is
fine for a decision-support prototype and keeps the maths dependency-free.
"""
from __future__ import annotations

import numpy as np

R_EARTH_KM = 6371.0
LAT_TS = -71.0
K0 = (1.0 + np.sin(np.deg2rad(abs(LAT_TS)))) / 2.0
_2RK = 2.0 * R_EARTH_KM * K0


def to_xy(lat, lon):
    """(lat, lon) in degrees -> (x, y) in km. Accepts scalars or arrays."""
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    rho = _2RK * np.tan(np.pi / 4.0 + np.deg2rad(lat) / 2.0)
    lam = np.deg2rad(lon)
    return rho * np.sin(lam), rho * np.cos(lam)


def to_latlon(x, y):
    """(x, y) in km -> (lat, lon) in degrees."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    rho = np.hypot(x, y)
    lat = np.rad2deg(2.0 * np.arctan(rho / _2RK) - np.pi / 2.0)
    lon = np.rad2deg(np.arctan2(x, y))
    return lat, lon


def scale_factor(lat):
    """Map scale factor k at latitude (projected distance / true distance)."""
    phi = np.deg2rad(np.asarray(lat, dtype=float))
    return 2.0 * K0 / (1.0 - np.sin(phi))


def east_north_basis(lon):
    """Unit vectors (in x/y map components) pointing geographic east and north.

    x = rho sin(lon), y = rho cos(lon), with rho growing toward the equator, so
    north (increasing latitude) is radially outward.
    """
    lam = np.deg2rad(np.asarray(lon, dtype=float))
    east = (np.cos(lam), -np.sin(lam))
    north = (np.sin(lam), np.cos(lam))
    return east, north


def en_to_xy(u_east, v_north, lon):
    """Rotate an (east, north) vector field into map (x, y) components."""
    (ex, ey), (nx, ny) = east_north_basis(lon)
    return u_east * ex + v_north * nx, u_east * ey + v_north * ny


def xy_to_en(vx, vy, lon):
    (ex, ey), (nx, ny) = east_north_basis(lon)
    return vx * ex + vy * ey, vx * nx + vy * ny


def bearing_deg(x0, y0, x1, y1):
    """True bearing (deg clockwise from north) of the segment p0 -> p1."""
    _, lon = to_latlon((x0 + x1) / 2.0, (y0 + y1) / 2.0)
    e, n = xy_to_en(x1 - x0, y1 - y0, lon)
    return (np.rad2deg(np.arctan2(e, n)) + 360.0) % 360.0

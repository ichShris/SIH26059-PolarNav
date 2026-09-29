"""Data access layer.

Everything the models consume goes through ``DataStore``. Today it is backed by the
synthetic ``World`` (see synth.py); the method signatures mirror what a real ingestion
pipeline provides, so wiring in OSI-SAF / NSIDC SIC, ERA5 / ECMWF winds and CMEMS
currents means re-implementing these methods on top of xarray readers.
"""
from __future__ import annotations

import threading
import time

import numpy as np
from scipy import ndimage

from . import config as C
from . import geo
from .synth import World, upsample2

WORLD_FILE = C.ARTIFACT_DIR / "world_v1.npz"


class DataStore:
    def __init__(self, masks, sic_u8, wind_truth, wind_fc):
        self.masks = masks
        self.world = World(masks)
        self._sic_u8 = sic_u8          # (days, N, N) uint8, SIC * 250
        self._wind_truth = wind_truth  # (days, 2, N/2, N/2) float16, m/s
        self._wind_fc = wind_fc        # (days, 3, 2, N/2, N/2) float16, forecasts issued each day
        self.n_days = sic_u8.shape[0]
        sl = self.world.ice_sl
        self.ice_blocked = masks["blocked"][sl]
        self.ice_coast = masks["coast_km"][sl]
        self.ice_lat, self.ice_lon = self.world.ice_lat, self.world.ice_lon

    # ----------------------------------------------------------------- build
    @classmethod
    def build(cls, log=print) -> "DataStore":
        t0 = time.time()
        masks = geo.build_masks()
        world = World(masks)
        log("simulating 4 years of daily sea ice and winds ...")
        sic, wind_truth = world.simulate_sic(
            progress=lambda d, n: log(f"  sea ice day {d}/{n}  ({time.time() - t0:.0f}s)"))
        log("generating 72 h forecast winds for every issue day ...")
        n = sic.shape[0]
        wind_fc = np.empty((n, C.FORECAST_DAYS, 2, C.ICE_N // 2, C.ICE_N // 2), dtype=np.float16)
        for d in range(n):
            for k in range(C.FORECAST_DAYS):
                wind_fc[d, k] = world.wind_grid_ice(d + 1 + k, issue_day=d, coarse=True)
            if d % 200 == 0:
                log(f"  forecast winds {d}/{n}  ({time.time() - t0:.0f}s)")
        sic_u8 = np.round(sic * 250).astype(np.uint8)
        C.ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(WORLD_FILE, sic_u8=sic_u8, wind_truth=wind_truth, wind_fc=wind_fc,
                            **{f"mask_{k}": v for k, v in masks.items()})
        log(f"world cached to {WORLD_FILE} in {time.time() - t0:.0f}s")
        return cls(masks, sic_u8, wind_truth, wind_fc)

    @classmethod
    def load(cls) -> "DataStore":
        z = np.load(WORLD_FILE)
        masks = {k[5:]: z[k] for k in z.files if k.startswith("mask_")}
        return cls(masks, z["sic_u8"], z["wind_truth"], z["wind_fc"])

    # ----------------------------------------------------------------- access
    def sic_truth(self, day: int) -> np.ndarray:
        return self._sic_u8[day].astype(np.float32) / 250.0

    def sic_obs(self, day: int) -> np.ndarray:
        """Satellite analysis (passive-microwave-like): truth + correlated retrieval error."""
        rng = np.random.default_rng(10_000 + day)
        err = ndimage.gaussian_filter(rng.standard_normal((C.ICE_N, C.ICE_N)), 1.5) * 0.12
        s = self.sic_truth(day)
        o = np.clip(s + err * (s > 0.02), 0.0, 1.0)
        o[self.ice_blocked] = 0.0
        return o.astype(np.float32)

    def wind_truth(self, day: int, coarse: bool = True) -> np.ndarray:
        w = self._wind_truth[day].astype(np.float32)
        return w if coarse else upsample2(w)

    def wind_forecast(self, issue_day: int, coarse: bool = True) -> np.ndarray:
        """(3, 2, n, n): daily-mean forecast winds for issue+1 .. issue+3."""
        w = self._wind_fc[issue_day].astype(np.float32)
        return w if coarse else upsample2(w)

    def thickness(self, sic: np.ndarray, day: int) -> np.ndarray:
        return self.world.thickness(sic, day)

    @staticmethod
    def embed(ice_field: np.ndarray, fill: float = 0.0) -> np.ndarray:
        """Place an ice-grid field into the full navigation grid."""
        out = np.full((C.NAV_N, C.NAV_N), fill, dtype=np.float32)
        sl = slice(C.ICE_OFF, C.ICE_OFF + C.ICE_N)
        out[sl, sl] = ice_field
        return out


_store: DataStore | None = None
_lock = threading.Lock()


def get_store(log=print) -> DataStore:
    global _store
    with _lock:
        if _store is None:
            _store = DataStore.load() if WORLD_FILE.exists() else DataStore.build(log=log)
        return _store

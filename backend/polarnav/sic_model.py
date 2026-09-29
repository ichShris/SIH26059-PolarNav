"""ConvLSTM sea-ice concentration forecaster (+24 h, +48 h, +72 h).

Architecture (≈90 k parameters, trains on CPU in minutes):

    7 days of [SIC analysis, analysis wind u/v, static land/coast, day-of-year]
        -> stride-2 conv encoder (160x160 -> 80x80)
        -> ConvLSTM (32 hidden channels) over the 7-day history
        -> fuse with ECMWF-style forecast winds for days +1..+3
        -> upsample, concat last analysis, predict SIC increments for 3 leads

The network predicts *changes* relative to persistence, which is the standard way
to make short-range sea-ice models beat persistence rather than relearn it.
"""
from __future__ import annotations

import json
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config as C
from .data import DataStore

MODEL_FILE = C.ARTIFACT_DIR / "sic_convlstm.pt"
METRICS_FILE = C.ARTIFACT_DIR / "sic_metrics.json"
H = C.HISTORY_DAYS
L = C.FORECAST_DAYS
WIND_SCALE = 15.0


class ConvLSTMCell(nn.Module):
    def __init__(self, cin: int, hid: int, k: int = 3):
        super().__init__()
        self.hid = hid
        self.conv = nn.Conv2d(cin + hid, 4 * hid, k, padding=k // 2)

    def forward(self, x, state):
        h, c = state
        i, f, o, g = torch.chunk(self.conv(torch.cat([x, h], 1)), 4, dim=1)
        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h = torch.sigmoid(o) * torch.tanh(c)
        return h, c


class SICForecaster(nn.Module):
    def __init__(self, hid: int = 32):
        super().__init__()
        self.hid = hid
        self.enc = nn.Sequential(nn.Conv2d(1, 12, 3, stride=2, padding=1), nn.GELU())
        self.cell = ConvLSTMCell(12 + 2 + 4, hid)
        self.fuse = nn.Sequential(nn.Conv2d(hid + 2 * L, hid, 3, padding=1), nn.GELU(),
                                  nn.Conv2d(hid, hid, 3, padding=1), nn.GELU())
        self.dec = nn.Sequential(nn.Conv2d(hid + 2, 24, 3, padding=1), nn.GELU(),
                                 nn.Conv2d(24, L, 3, padding=1))

    def forward(self, sic_hist, wind_hist, static_c, wind_fc, static_f):
        """
        sic_hist  (B, H, N, N)       analysis SIC
        wind_hist (B, H, 2, n, n)    analysis winds (coarse, scaled)
        static_c  (B, 4, n, n)       land, coast, sin doy, cos doy on coarse grid
        wind_fc   (B, L*2, n, n)     forecast winds for the next L days (coarse, scaled)
        static_f  (B, 1, N, N)       ocean mask on the fine grid
        """
        b, _, n2, _ = static_c.shape
        h = sic_hist.new_zeros(b, self.hid, n2, n2)
        c = torch.zeros_like(h)
        for t in range(sic_hist.shape[1]):
            e = self.enc(sic_hist[:, t:t + 1])
            h, c = self.cell(torch.cat([e, wind_hist[:, t], static_c], 1), (h, c))
        z = self.fuse(torch.cat([h, wind_fc], 1))
        z = F.interpolate(z, scale_factor=2, mode="bilinear", align_corners=False)
        last = sic_hist[:, -1:]
        delta = self.dec(torch.cat([z, last, static_f], 1))
        return torch.clamp(last + delta, 0.0, 1.0) * static_f


# ---------------------------------------------------------------------------- tensors
class _LazyDays:
    """Array-like day index that computes fields on demand (supports [d] and [list])."""

    def __init__(self, fn):
        self.fn = fn
        self.cache: dict[int, np.ndarray] = {}

    def _get(self, d: int) -> np.ndarray:
        if d not in self.cache:
            if len(self.cache) > 64:
                self.cache.clear()
            self.cache[d] = self.fn(int(d)).astype(np.float16)
        return self.cache[d]

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            idx = list(range(idx.start, idx.stop, idx.step or 1))
        if isinstance(idx, (list, tuple, range)):
            return np.stack([self._get(d) for d in idx])
        return self._get(idx)


class Batcher:
    """Assembles model inputs for any issue day from the DataStore."""

    def __init__(self, store: DataStore, precompute: bool = True):
        self.s = store
        n = store.n_days
        if precompute:  # training touches every day many times
            self.obs = np.stack([store.sic_obs(d) for d in range(n)]).astype(np.float16)
            self.truth = np.stack([store.sic_truth(d) for d in range(n)]).astype(np.float16)
        else:  # inference only needs a 7-day window
            self.obs = _LazyDays(store.sic_obs)
            self.truth = _LazyDays(store.sic_truth)
        land_c = store.ice_blocked[::2, ::2].astype(np.float32)
        coast_c = np.clip(store.ice_coast[::2, ::2] / 2000.0, 0, 2).astype(np.float32)
        self.land_c, self.coast_c = land_c, coast_c
        self.ocean_f = (~store.ice_blocked).astype(np.float32)[None]

    def inputs(self, days: list[int]):
        sh, wh, sc, wf, sf = [], [], [], [], []
        for t in days:
            hist = range(t - H + 1, t + 1)
            sh.append(self.obs[list(hist)].astype(np.float32))
            wh.append(np.stack([self.s.wind_truth(d) for d in hist]) / WIND_SCALE)
            doy = C.index_day(t).timetuple().tm_yday
            a = 2 * math.pi * doy / 365.25
            ones = np.ones_like(self.land_c)
            sc.append(np.stack([self.land_c, self.coast_c, ones * math.sin(a), ones * math.cos(a)]))
            wf.append(self.s.wind_forecast(t).reshape(L * 2, *self.land_c.shape) / WIND_SCALE)
            sf.append(self.ocean_f)
        f = lambda xs: torch.from_numpy(np.stack(xs).astype(np.float32))
        return f(sh), f(wh), f(sc), f(wf), f(sf)

    def targets(self, days: list[int]):
        return torch.from_numpy(np.stack([self.truth[t + 1:t + 1 + L] for t in days]).astype(np.float32))


def split_days(n_days: int):
    tr_end, va_end = C.day_index(C.TRAIN_END), C.day_index(C.VAL_END)
    ok = range(H - 1, n_days - L)
    train = [t for t in ok if t + L <= tr_end]
    val = [t for t in ok if tr_end < t and t + L <= va_end]
    test = [t for t in ok if t > va_end]
    return train, val, test


# ---------------------------------------------------------------------------- metrics
def climatology(batcher: Batcher, train_days: list[int]) -> np.ndarray:
    """Mean truth SIC for each day-of-year (±5 day window) over the training period."""
    doys = np.array([C.index_day(d).timetuple().tm_yday for d in range(batcher.truth.shape[0])])
    train_set = np.zeros(len(doys), bool)
    train_set[: max(train_days) + L + 1] = True
    clim = np.zeros((367, C.ICE_N, C.ICE_N), np.float32)
    for doy in range(1, 367):
        dd = np.abs(((doys - doy + 182) % 365) - 182) <= 5
        clim[doy] = batcher.truth[dd & train_set].astype(np.float32).mean(0)
    return clim


def skill(pred: np.ndarray, truth: np.ndarray, ocean: np.ndarray) -> dict:
    """RMSE (% SIC) over the active ice zone and IIEE (thousand km^2) at the 15% edge."""
    zone = ocean & ((truth > 0.01) | (pred > 0.01))
    rmse = float(np.sqrt(np.mean((pred[zone] - truth[zone]) ** 2)) * 100) if zone.any() else 0.0
    iiee = float(((pred > 0.15) != (truth > 0.15))[ocean].sum() * C.CELL_KM ** 2 / 1e3)
    return {"rmse": rmse, "iiee": iiee}


def evaluate(model, batcher: Batcher, days: list[int], clim: np.ndarray) -> dict:
    ocean = ~batcher.s.ice_blocked
    acc = {k: [[] for _ in range(L)] for k in ("model", "persistence", "climatology")}
    model.eval()
    with torch.no_grad():
        for i in range(0, len(days), 8):
            chunk = days[i:i + 8]
            pred = model(*batcher.inputs(chunk)).numpy()
            tgt = batcher.targets(chunk).numpy()
            for j, t in enumerate(chunk):
                pers = batcher.obs[t].astype(np.float32)
                for k in range(L):
                    doy = C.index_day(t + 1 + k).timetuple().tm_yday
                    acc["model"][k].append(skill(pred[j, k], tgt[j, k], ocean))
                    acc["persistence"][k].append(skill(pers, tgt[j, k], ocean))
                    acc["climatology"][k].append(skill(clim[doy], tgt[j, k], ocean))
    out = {}
    for name, per_lead in acc.items():
        out[name] = [{m: float(np.mean([r[m] for r in rows])) for m in ("rmse", "iiee")} for rows in per_lead]
    return out


# ---------------------------------------------------------------------------- train
def train(store: DataStore, minutes: float = 8.0, batch: int = 8, log=print) -> dict:
    torch.manual_seed(0)
    torch.set_num_threads(max(1, torch.get_num_threads()))
    batcher = Batcher(store)
    train_days, val_days, test_days = split_days(store.n_days)
    model = SICForecaster()
    n_params = sum(p.numel() for p in model.parameters())
    log(f"ConvLSTM params: {n_params:,}; train {len(train_days)} / val {len(val_days)} / test {len(test_days)} issue days")
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    ocean = torch.from_numpy(batcher.ocean_f)
    rng = np.random.default_rng(0)
    t0, it, losses = time.time(), 0, []
    budget = minutes * 60
    while time.time() - t0 < budget:
        lr = 2e-3 * 0.5 * (1 + math.cos(math.pi * min((time.time() - t0) / budget, 1.0)))
        for g in opt.param_groups:
            g["lr"] = max(lr, 5e-5)
        days = list(rng.choice(train_days, batch, replace=False))
        model.train()
        pred = model(*batcher.inputs(days))
        tgt = batcher.targets(days)
        # weight cells that carry ice so the vast open ocean doesn't dominate
        w = ocean * (1.0 + 4.0 * ((tgt > 0.02) | (pred.detach() > 0.02)).float())
        loss = (w * (pred - tgt) ** 2).sum() / w.sum()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
        it += 1
        if it % 25 == 0:
            log(f"  iter {it:4d}  loss {np.mean(losses[-25:]):.5f}  ({time.time() - t0:.0f}s)")
    C.ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_FILE)
    log("evaluating on validation (2024 H2) and held-out test year (2025) ...")
    clim = climatology(batcher, train_days)
    metrics = {
        "params": n_params,
        "iterations": it,
        "train_minutes": round((time.time() - t0) / 60, 1),
        "splits": {"train": [str(C.index_day(train_days[0])), str(C.index_day(train_days[-1]))],
                   "val": [str(C.index_day(val_days[0])), str(C.index_day(val_days[-1]))],
                   "test": [str(C.index_day(test_days[0])), str(C.index_day(test_days[-1]))]},
        "val": evaluate(model, batcher, val_days[::3], clim),
        "test": evaluate(model, batcher, test_days[::3], clim),
        "leads_h": [24 * (k + 1) for k in range(L)],
    }
    METRICS_FILE.write_text(json.dumps(metrics, indent=2))
    for split in ("val", "test"):
        for k in range(L):
            m = metrics[split]
            log(f"  {split} +{24 * (k + 1)}h  RMSE model {m['model'][k]['rmse']:.2f}  pers {m['persistence'][k]['rmse']:.2f}"
                f"  clim {m['climatology'][k]['rmse']:.2f} | IIEE model {m['model'][k]['iiee']:.0f}"
                f"  pers {m['persistence'][k]['iiee']:.0f}  clim {m['climatology'][k]['iiee']:.0f} (10^3 km2)")
    return metrics


# ---------------------------------------------------------------------------- inference
class Forecaster:
    def __init__(self, store: DataStore):
        self.store = store
        self.batcher = Batcher(store, precompute=False)
        self.model = SICForecaster()
        self.model.load_state_dict(torch.load(MODEL_FILE, map_location="cpu", weights_only=True))
        self.model.eval()
        self.metrics = json.loads(METRICS_FILE.read_text()) if METRICS_FILE.exists() else None

    def forecast(self, day: int) -> np.ndarray:
        """(3, N, N) forecast SIC for day+1 .. day+3 issued on `day`."""
        with torch.no_grad():
            return self.model(*self.batcher.inputs([day]))[0].numpy()

    def daily_skill(self, day: int, pred: np.ndarray) -> list[dict]:
        ocean = ~self.store.ice_blocked
        pers = self.batcher.obs[day].astype(np.float32)
        out = []
        for k in range(L):
            if day + 1 + k >= self.store.n_days:
                break
            truth = self.store.sic_truth(day + 1 + k)
            out.append({"lead_h": 24 * (k + 1), "model": skill(pred[k], truth, ocean),
                        "persistence": skill(pers, truth, ocean)})
        return out

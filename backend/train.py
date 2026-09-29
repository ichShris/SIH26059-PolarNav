"""One-time setup: build the environment dataset, train the ConvLSTM, fit the hybrid drift model.

    python train.py                 # everything (≈ 3 min data + --minutes of training + ≈ 2 min drift)
    python train.py --minutes 4     # shorter ConvLSTM budget
    python train.py --only drift    # refit only the iceberg drift correction
"""
from __future__ import annotations

import argparse
import sys
import time


def log(msg: str) -> None:
    print(msg, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=12.0, help="ConvLSTM wall-clock training budget")
    ap.add_argument("--only", choices=["sic", "drift"], default=None)
    args = ap.parse_args()

    t0 = time.time()
    from polarnav.data import get_store
    store = get_store(log=log)
    log(f"dataset ready: {store.n_days} days ({time.time() - t0:.0f}s)")

    if args.only in (None, "sic"):
        from polarnav import sic_model
        log("training ConvLSTM sea-ice forecaster ...")
        sic_model.train(store, minutes=args.minutes, log=log)

    if args.only in (None, "drift"):
        from polarnav.drift import DriftEngine
        log("fitting hybrid iceberg drift correction ...")
        DriftEngine(store).fit_hybrid(log=log)

    log(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    sys.exit(main())

"""Start PolarNav: API + dashboard on http://127.0.0.1:8000

    python run.py [--port 8000] [--host 127.0.0.1]

First run builds the synthetic dataset (~3 min) if backend/artifacts is empty.
Model weights and metrics are produced by backend/train.py.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "backend"))

import uvicorn  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    from polarnav.api import svc
    svc()  # load data + models before accepting requests
    uvicorn.run("polarnav.api:app", host=a.host, port=a.port, log_level="info")

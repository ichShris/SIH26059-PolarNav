"""Low-bandwidth shore -> ship sync.

Heavy models run ashore; the ship receives a compact binary bundle and renders it in
the offline dashboard. Encoding: SIC as 1 % steps (uint8) with the 3 forecast slices
delta-coded against the analysis, POLARIS RIO as int8 per slice, iceberg tracks as
int16 hectometre offsets, route as a waypoint list, all zlib-compressed. The same
fields as float32 arrays + JSON are reported for comparison.
"""
from __future__ import annotations

import json
import struct
import zlib

import numpy as np

LINKS_KBPS = {"Iridium Certus 100 (88 kbps)": 88, "FleetBroadband (432 kbps)": 432, "Iridium SBD-class (2.4 kbps)": 2.4}


def pack(date: str, ice_class: str, sic_slices: list[np.ndarray], rio_slices: list[np.ndarray],
         bergs: list[dict], waypoints: list[dict]) -> tuple[bytes, dict]:
    sic_q = [np.round(np.clip(s, 0, 1) * 100).astype(np.int16) for s in sic_slices]
    sic_parts = [sic_q[0].astype(np.uint8)] + [(sic_q[k] - sic_q[0]).astype(np.int8) for k in range(1, len(sic_q))]
    rio_parts = [np.clip(np.round(r), -127, 127).astype(np.int8) for r in rio_slices]
    berg_blob = bytearray()
    for b in bergs:
        tr = np.asarray(b["forecast"])[::2]  # 6-hourly
        ell = [e["a"] for e in b["ellipses"]]
        berg_blob += struct.pack("<6sH", b["id"].encode()[:6], len(tr))
        origin = tr[0]
        berg_blob += struct.pack("<ff", *origin)
        berg_blob += np.round((tr - origin) * 10).astype(np.int16).tobytes()
        berg_blob += np.round(np.asarray(ell[:len(tr)]) * 10).astype(np.uint16).tobytes()
    header = json.dumps({"v": 1, "date": date, "ice_class": ice_class, "grid": list(sic_slices[0].shape),
                         "slices": len(sic_slices), "bergs": len(bergs),
                         "wp": [[w["name"], w["lat"], w["lon"], w["t_h"]] for w in waypoints]},
                        separators=(",", ":")).encode()
    body = b"".join(p.tobytes() for p in sic_parts) + b"".join(p.tobytes() for p in rio_parts) + bytes(berg_blob)
    payload = struct.pack("<I", len(header)) + header + body
    packed = zlib.compress(payload, 9)

    raw_arrays = sum(s.astype(np.float32).nbytes for s in sic_slices) + sum(r.astype(np.float32).nbytes for r in rio_slices)
    raw_json = len(json.dumps({"bergs": bergs, "waypoints": waypoints}).encode())
    raw = raw_arrays + raw_json
    stats = {
        "raw_bytes": raw,
        "packed_bytes": len(packed),
        "ratio": round(raw / len(packed), 1),
        "contents": {"sic_slices": len(sic_slices), "rio_slices": len(rio_slices), "bergs": len(bergs),
                     "waypoints": len(waypoints)},
        "transfer_s": {name: {"raw": round(raw * 8 / (k * 1000), 1), "packed": round(len(packed) * 8 / (k * 1000), 1)}
                       for name, k in LINKS_KBPS.items()},
    }
    return packed, stats

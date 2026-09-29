"""Download Natural Earth 1:50m land + Antarctic ice shelves and keep the Southern Hemisphere part.

Natural Earth is public domain. Run once; the filtered files are committed under backend/data/.
    python scripts/fetch_coastline.py [--src DIR_WITH_LOCAL_GEOJSON]
"""
from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

BASE = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/"
FILES = {
    "ne_50m_land.geojson": "ne_50m_land_south.geojson",
    "ne_50m_antarctic_ice_shelves_polys.geojson": "ne_50m_antarctic_ice_shelves_south.geojson",
}
OUT_DIR = Path(__file__).resolve().parents[1] / "backend" / "data"
LAT_CUT = -12.0


def _load(name: str, src: Path | None) -> dict:
    if src and (src / name).exists():
        return json.loads((src / name).read_text(encoding="utf-8"))
    with urllib.request.urlopen(BASE + name, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))


def _filter(gj: dict) -> dict:
    feats = []
    for f in gj["features"]:
        g = f["geometry"]
        polys = [g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]
        keep = [[[[round(x, 4), round(y, 4)] for x, y, *_ in ring] for ring in poly]
                for poly in polys if any(pt[1] < LAT_CUT for pt in poly[0])]
        if keep:
            feats.append({"type": "Feature", "properties": {},
                          "geometry": {"type": "MultiPolygon", "coordinates": keep}})
    return {"type": "FeatureCollection", "features": feats}


def _globe(land: dict, shelves: dict) -> dict:
    """World land + Antarctic ice shelves as compact lon/lat rings for the 3-D globe view.

    Natural Earth closes Antarctica with a detour down the 180 deg meridian to the pole and
    ~250 vertices along lat -90. On a sphere those are all the same point; they make polygon
    fills misbehave when the globe is viewed from above the pole, so they are dropped and
    the ring simply closes across the antimeridian at ~84 S.

    Two levels of detail: 'land'/'shelf' (~0.12 deg) for a still globe, 'land_lo'/'shelf_lo'
    (~0.6 deg, small islands dropped) for smooth rendering while it rotates or zooms.
    """
    def rings(gj, tol_deg: float, min_extent: float):
        out = []
        for f in gj["features"]:
            g = f["geometry"]
            polys = [g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]
            for poly in polys:
                kept_poly = []
                for ri, ring in enumerate(poly):
                    # drop the artificial pole detour: the pole itself and the +/-180 meridian legs
                    pts = [pt[:2] for pt in ring
                           if pt[1] > -89.9 and not (pt[1] < -84.0 and abs(abs(pt[0]) - 180.0) < 1e-6)]
                    if len(pts) < 4:
                        continue
                    xs = [p[0] for p in pts]
                    ys = [p[1] for p in pts]
                    if ri == 0 and max(max(xs) - min(xs), max(ys) - min(ys)) < min_extent:
                        break  # drop tiny islands (and their holes) at this level of detail
                    kept = [pts[0]]
                    for pt in pts[1:-1]:
                        lx, ly = kept[-1]
                        if abs(pt[0] - lx) + abs(pt[1] - ly) >= tol_deg:
                            kept.append(pt)
                    kept.append(pts[0])
                    if len(kept) >= 4:
                        kept_poly.append([[round(x, 2), round(y, 2)] for x, y in kept])
                if kept_poly:
                    out.append(kept_poly)
        return out
    return {"land": rings(land, 0.12, 0.0), "shelf": rings(shelves, 0.12, 0.0),
            "land_lo": rings(land, 0.6, 0.8), "shelf_lo": rings(shelves, 0.6, 0.8)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=None)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = {}
    for src_name, dst_name in FILES.items():
        raw[src_name] = _load(src_name, args.src)
        out = _filter(raw[src_name])
        (OUT_DIR / dst_name).write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
        print(f"{dst_name}: {len(out['features'])} features")
    globe = _globe(*raw.values())
    (OUT_DIR / "globe_land.json").write_text(json.dumps(globe, separators=(",", ":")), encoding="utf-8")
    n = {k: sum(len(r) for p in v for r in p) for k, v in globe.items()}
    print(f"globe_land.json: {len(globe['land'])} land polygons ({n['land']} vertices), "
          f"low-detail {len(globe['land_lo'])} polygons ({n['land_lo']} vertices)")


if __name__ == "__main__":
    main()

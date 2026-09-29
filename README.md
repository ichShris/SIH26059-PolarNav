# PolarNav — AI-enabled Antarctic sea-ice, iceberg and navigation decision support

Working prototype for **SIH 2026 · Problem Statement 26059** (MoES / NCPOR), team *Git Gud* (SIH26-A0H-T025).

PolarNav forecasts Antarctic sea-ice concentration 72 h ahead with a ConvLSTM, predicts iceberg trajectories with a
hybrid physics-AI ensemble, rates every grid cell against the IMO POLARIS code for the ship's ice class, and plans a
fuel-optimal, iceberg-aware route. Results reach the ship as a ~29 KB bundle that an offline dashboard runs from.

## Quick start

Requires Python 3.11+ and Node 18+.

```bash
python -m venv .venv
.venv/Scripts/python -m pip install numpy scipy fastapi "uvicorn[standard]" pydantic pytest httpx
.venv/Scripts/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
cd frontend && npm install && npm run build && cd ..
.venv/Scripts/python run.py            # open http://127.0.0.1:8000
```

(On macOS/Linux use `.venv/bin/python`.) The trained ConvLSTM weights and all metrics are in `backend/artifacts/`.
On first start the service regenerates the 4-year environment dataset (`world_v1.npz`, ~3 min, deterministic seed).

| Task | Command (from `backend/`) |
|---|---|
| Rebuild data + retrain everything | `python train.py` (≈ 3 min data + 12 min ConvLSTM on CPU + 1 min drift) |
| Refit only the iceberg drift correction | `python train.py --only drift` |
| Season route benchmark | `python benchmark_routes.py` |
| Tests | `python -m pytest tests` (15 tests) |
| Frontend dev server with hot reload | `cd frontend && npm run dev` (proxies `/api` to :8000) |

## Demo walkthrough (5 min)

0. **Globe** (default view) — a 3-D globe with every continent: drag to rotate (it keeps gliding when you flick it), scroll
   to zoom, *↻ Spin* to auto-rotate, *South Pole* / *Route* / *Reset* to fly to a view. All layers, routes and icebergs are
   drawn on the globe; switch to *Polar chart* for the flat polar stereographic navigation chart. The globe is rendered
   with WebGL (a fragment shader fills ocean, land and the sea-ice raster per pixel; routes, icebergs and coastlines are
   vector overlays), so rotation and deep zoom run at display frame rate (~1 ms of script per frame).
1. **Default scenario** — 20 Nov 2025, vessel at 58°S 14°E heading for the Maitri offload point, PC5 ship. The cyan line is
   the PolarNav route, the dashed orange line is the conventional shortest route on today's ice chart.
2. **Route options** (left) — every plan returns four routes: *Recommended* (your weights), *Fastest*, *Least ice
   exposure* and the *Conventional* shortest route. Click one to select it (bold on the map; vessel, timeline and
   waypoints follow it). Routes re-plan automatically ~0.5 s after any setting changes. When a setting cannot move the
   track today (e.g. no iceberg near the corridor, no low-RIO ice), the panel says why.
3. **Timeline** (bottom) — press ▶ or jump to +24/48/72 h: the vessel, the drifting icebergs and the sea-ice field all move to
   that time. Route segments turn amber where the ship is in elevated-risk ice (speed-limited per POLARIS).
4. **Layers** — *POLARIS risk* for the chosen ice class (change *Ice class* at the top to see the map and route react),
   *Forecast error*: at **Now** it shows yesterday's +24 h forecast minus today's satellite analysis — a check that works in
   real time; at +24/48/72 h it shows forecast minus the verifying truth, which is only possible in hindcast (2025 is a
   held-out year). *Verify vs truth* overlays true iceberg tracks the same way.
5. **Iceberg hazards** tab — click a berg: its 30-member ensemble, 2σ ellipses, physics-only track (purple) and truth (green).
6. **AI models** tab — ConvLSTM vs persistence vs climatology, drift-model error, and the season routing benchmark.
7. **SAR detect** tab — CFAR iceberg detection on Sentinel-1-like scenes, scored against ground truth.
8. **Edge sync** tab — bundle size and transfer time per satellite link; tick *Simulate satellite-link blackout* and
   recompute: the dashboard keeps running from its cached bundle.

## Architecture

```
 Data layer (backend/polarnav/data.py)          AI / physics core                          Decision layer
 ┌──────────────────────────────┐   ┌─────────────────────────────────────┐   ┌────────────────────────────────┐
 │ SIC analysis  (≈ OSI-SAF/NSIDC)│──▶│ sic_model.py  ConvLSTM +24/48/72 h  │──▶│ polaris.py  RIO per ice class  │
 │ Winds + fcst  (≈ ERA5 / ECMWF) │──▶│ drift.py      physics ensemble +    │──▶│ fuel.py     h-v curve, fuel/km │
 │ Currents      (≈ CMEMS)        │   │               learned correction    │   │ routing.py  time-dependent A*  │
 │ SAR scenes    (≈ Sentinel-1)   │──▶│ sar.py        CA-CFAR + discrimin.  │   │ edge_sync.py 29 KB ship bundle │
 └──────────────────────────────┘   └─────────────────────────────────────┘   └───────────────┬────────────────┘
                                                        api.py (FastAPI)  ◀─────────────────────────┘
                                                              │
                                       frontend/ React dashboard (offline cache, polar stereographic chart)
```

| Module | What it does |
|---|---|
| `synth.py` | Reproducible Southern Ocean "digital twin": westerlies, coastal easterlies, katabatic outflow and ~1,900 transient cyclones; forecast winds whose error grows with lead time (~100 km cyclone position error per day); ACC, coastal current, Weddell/Ross gyres; wind-driven sea-ice dynamics on a realistic seasonal cycle (≈2 M km² Feb minimum, ≈21 M km² Sep maximum). |
| `sic_model.py` | 86 k-parameter ConvLSTM: 7 days of SIC + winds → stride-2 encoder → ConvLSTM → fused with 3 days of forecast winds → SIC increments for +24/+48/+72 h. Trained 2022-01 → 2024-06, validated 2024-H2, tested on all of 2025. |
| `drift.py` | Iceberg momentum balance (air/water form drag, Coriolis, sea-surface slope, pack-ice capture) with a semi-implicit integrator, 30-member ensemble (forecast-wind members, drag perturbations, SAR geolocation error) and a complex-valued ridge correction learned from historical tracks. |
| `polaris.py` | IMO MSC.1/Circ.1519 RIO = Σ Cᵢ·RIVᵢ with the RIV table for PC1–PC7, IA Super–IC and no ice class; normal / elevated / special-consideration levels. |
| `fuel.py` | Cube-law open-water power + ice resistance calibrated to 3 kn in rated level ice; POLARIS speed limits in elevated-risk ice; fuel from SFOC, CO₂ from the IMO MGO factor. |
| `routing.py` | 16-connected, time-dependent A* on a 50 km polar stereographic grid: ice field valid at the ship's ETA, moving iceberg exclusion zones that grow with forecast uncertainty (configurable clearance), hard POLARIS limits, soft penalties for RIO below a safety buffer of 10 and for ice exposure. Returns recommended, fastest and least-ice-exposure options plus the conventional baseline (shortest path on today's ice chart). |
| `sar.py` | Sentinel-1-like scenes (4-look gamma speckle, wind-modulated clutter, pack ice) + two-stage detector: CA-CFAR prescreen, size/backscatter discrimination. |
| `edge_sync.py` | Quantised, delta-coded, zlib-compressed shore→ship bundle. |

## Measured results (held-out test year 2025)

| Component | Metric | PolarNav | Baseline(s) |
|---|---|---|---|
| Sea-ice forecast | RMSE in the ice zone, +24 / +48 / +72 h | **4.0 / 5.1 / 5.8 %** | persistence 4.4 / 6.0 / 7.2 %, climatology 7.3 % |
| Sea-ice forecast | Ice-edge error (IIEE) at +72 h | **565 k km²** | persistence 701, climatology 695 |
| Iceberg drift | 72 h position error, mean / median (240 tracks) | **6.2 / 3.7 km** | physics-only 14.2 / 10.3, persistence 33.8 / 25.6 |
| SAR detection | recall / precision (40 scenes, Pfa 1e-5) | 99.7 % / 99.3 % | single-stage CFAR: ~39 % precision on a sample scene |
| Routing (54 PC5 ice passages to Maitri & Bharati, Nov–Feb) | fuel saving vs conventional route | **median 4.4 %, mean 7.4 %, 10.5 % in heavy ice, up to 24 %** | — |
| Routing | voyages passing an iceberg < 10 nm (verified against truth) | **1** | 5 |
| Edge sync | ship bundle | **~29 KB, 33× smaller**, 2.7 s over Iridium Certus 100 | 952 KB raw |

## Real-time vs hindcast verification

Forecast error *for the future* cannot be computed in real time — the truth has not happened yet. Two checks exist:

| Check | Needs future data? | Where |
|---|---|---|
| Forecasts issued 1–3 days ago, valid today, vs **today's satellite analysis** | No — runs live every day | *Forecast error* layer at **Now**; *AI models → Real-time forecast check* |
| Forecast vs verifying truth at +24/48/72 h | Yes — hindcast only | *Forecast error* layer at +24/48/72 h; skill charts |

In live use the ensemble spread (iceberg ellipses) is the forward-looking uncertainty estimate.

## Limitations — read before presenting

* **All data is synthetic.** The prototype runs offline without Copernicus / ECMWF / NSIDC credentials, so every feed comes
  from `synth.py`. The environment is physically structured (realistic seasonal extent, cyclone climatology, free-drift ice
  physics), but the scores above are scores *on that environment*, not on the real Southern Ocean. Real-world 72 h iceberg
  errors are typically larger (tens of km) and real SAR is harder (sea state, bergy bits, ships).
* **The deck's ">15 % fuel reduction" is only reached in heavy early-season ice** in our benchmark; the honest season-wide
  figure is a ~4–7 % median/mean saving over the ice passage, ~10 % in heavy ice. Recommend quoting the distribution.
* **YOLOv8 is not implemented**; the CFAR + discrimination stage is the detection front-end that would feed it once labelled
  Sentinel-1 chips are available. The ConvLSTM is deliberately small (CPU-trainable in 12 min); ice thickness is a
  diagnostic from concentration and pack depth, not a separate forecast.
* The POLARIS RIV table is transcribed from MSC.1/Circ.1519 Table 1.3 — verify against the circular before operational use.
  The tool is advisory only; the master retains full responsibility.
* 50 km grid: adequate for passage planning, too coarse for final approach through leads or fast ice.

## Swapping in real data

Only `backend/polarnav/data.py` has to change — the models consume its methods:

| `DataStore` method | Real source |
|---|---|
| `sic_obs(day)` | OSI-SAF OSI-401/408 or NSIDC-0051/0081 daily SIC (regrid to the 50 km EPSG:3031 grid) |
| `wind_truth(day)` / `world.wind(...)` | ERA5 10 m winds (Copernicus CDS) |
| `wind_forecast(issue_day)`, ensemble members | ECMWF HRES / ENS open data |
| `world.current(...)` | CMEMS GLOBAL_ANALYSISFORECAST_PHY surface currents |
| `thickness(...)` | CryoSat-2/SMOS merged thickness or CMEMS sea-ice thickness |
| iceberg positions | Sentinel-1 EW GRD → `sar.detect` (and later YOLOv8), or the BYU/NIC Antarctic iceberg database for tracks to refit the drift correction |

## API

`GET /api/meta` · `GET /api/coastline` · `GET /api/scenario?date=YYYY-MM-DD` · `GET /api/polaris?date=&ice_class=` ·
`POST /api/route` · `POST /api/sync/bundle` · `GET /api/sar?seed=&pfa=` — interactive docs at `/docs`.

Coastline: Natural Earth 1:50m land and Antarctic ice shelves (public domain), filtered by `scripts/fetch_coastline.py`.

# welfare-rs — welfare-oriented activity-travel simulation

An agent-based model of how recommender systems shape activity-travel behaviour
and traveller **welfare**, simulated on a real OpenStreetMap street network. It ships as three separable layers:

```
welfare_rs/   # the importable simulation engine (pip-installable)
backend/      # FastAPI service: runs the simulation, including road network and agent dynamics
frontend/     # React + TypeScript + deck.gl/MapLibre client (Vite)
data/         # persona CSVs + data-util scripts
```

The browser never holds the whole run: it streams only the agent **positions**
in the current time window + map viewport, and fetches a trip's full **street
geometry** on demand (when you click a moving dot). This lets the visualization
scale to large populations / many days without falling over.

---

## 1. Install

Requires Python ≥ 3.10 and (for the frontend) Node ≥ 18.

```bash
# from the repo root — engine + backend, editable
pip install -e .[server]

# or everything used in this repo (engine + backend + paper notebook)
pip install -r requirements.txt
```

The engine pulls `osmnx`, which brings `geopandas`/`pandas`/`shapely`. The first
run for an area **downloads and caches** that area's OSM network under `.cache/`
(set `WELFARE_RS_CACHE` to relocate it); subsequent runs are fully local.

> **Env note:** if you see `A module compiled using NumPy 1.x cannot be run in
> NumPy 2.x … _ARRAY_API not found`, that's a pre-existing mismatch between
> `pyarrow` (built against NumPy 1.x) and a NumPy 2.x environment. It's a warning,
> not a failure. Fix it with `pip install -U pyarrow` or `pip install "numpy<2"`.

## 2. Run the app (development)

Two processes. **Terminal A — backend:**

```bash
cd backend
uvicorn main:app --reload --port 8000
```

**Terminal B — frontend (Vite dev server, proxies `/api` to :8000):**

```bash
cd frontend
npm install
npm run dev          # http://localhost:5173
```

Open http://localhost:5173, pick an area, set options, press **Run**.

## 3. Run the app (single server, production-style)

Build the frontend once; FastAPI then serves it as static files at `/`:

```bash
cd frontend && npm install && npm run build      # -> frontend/dist
cd ../backend && uvicorn main:app --port 8000     # http://localhost:8000
```

## 4. Use the engine as a library

`welfare_rs` has no web or visualization dependency. Import and drive it directly:

```python
from welfare_rs import Simulation, build_road_network, params

net = build_road_network("nyc_manhattan")              # cached after first download
sim = Simulation(
    num_agents=80, seed=42, road_network=net,
    persona_csv_path=params.NYC_PERSONA_CSV_PATH,
    disabled_modes=("transit",),
)
day_summaries = sim.run_days(3)
print(sim.summarize())
```


## 5. HTTP API (under `/api`)

| Method | Path | Purpose |
|---|---|---|
| GET  | `/api/cities` | areas, treatments, default config |
| POST | `/api/runs` | start (or hit cached) run → `run_id` |
| GET  | `/api/runs/{id}` | run summary + view/bounds/metrics |
| GET  | `/api/runs/{id}/events` | SSE per-day progress |
| GET  | `/api/runs/{id}/timeline?t0&t1&bbox` | windowed, viewport-culled trip/stay keyframes |
| GET  | `/api/runs/{id}/positions?t&bbox` | exact-instant positions (interpolated on real routes) |
| GET  | `/api/runs/{id}/trips/{trip_id}/geometry` | **full** street polyline for one trip (on demand) |
| GET  | `/api/runs/{id}/pois?bbox` | POIs inside the viewport |

Interactive docs at `/docs` (FastAPI/Swagger). `bbox` is `south,west,north,east`.

## 6. POI data

The sim uses a small filtered CSV of NYC leisure POIs at
`~/.cache/welfare_rs/pois/nyc_leisure_pois.csv` (~4 MB). It is produced once from
a large raw Placekey/NAICS dump (`poi_children_merged_by_wkt.csv`, ~375 MB) by
`data/filter_nyc_pois.py`. **You almost never need to distribute the raw file** —
ship the small filtered CSV (commit it, or attach it as a GitHub Release asset).
If you ever must distribute the raw 375 MB file, don't put it in normal git
(GitHub blocks >100 MB/file): use a GitHub Release asset, Git LFS, or an external
host (Zenodo gives a citable DOI, nice for the paper).

Without the POI file the app falls back to synthetic POIs placed on the network
(the "Real NYC POIs" toggle disables itself).

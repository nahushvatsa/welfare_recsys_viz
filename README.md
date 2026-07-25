# welfare-rs — welfare-oriented activity-travel simulation

An agent-based model of how recommender systems shape activity-travel behaviour
and traveller **welfare**, simulated on a real OpenStreetMap street network. It ships as three separable layers:

```
welfare_rs/   # the importable simulation engine (pip-installable)
backend/      # FastAPI service: runs the simulation, including road network and agent dynamics
frontend/     # React + TypeScript + deck.gl/MapLibre client (Vite)
data/         # persona CSVs + data-util scripts (+ gitignored local data)
scripts/      # warm_metros.py — prefetch all metro road networks
```

## Two-layer metro geography

The app simulates **8 US metros** (NYC, Seattle, SF Bay Area, Chicago, Houston,
DC, Miami, LA). Each is ONE road network thought of as two layers:

* **Core** — the principal city (Manhattan for NYC; SF + Oakland for the Bay
  Area), downloaded at full drive detail. Leisure **POIs and work locations
  exist only here**; leisure routing is street-level.
* **Shell** — the surrounding counties that together send **90% of the
  commuters** into the core (LODES-style flows), reduced to the arterial
  skeleton (motorway/trunk/primary/secondary). Agent **homes** are sampled
  here (and in the core), weighted by each county's commuter count. Dropping
  residential capillaries outside the core is the standard MPO travel-model
  abstraction and keeps metro graphs tractable.

Cars route on **fastest network paths** (per-edge OSM speeds), so freeway
commutes behave like freeway commutes; congestion and weather scale that time.

**Data policy: all data lives on disk (or, later, the lab server's Postgres) —
never in git.** The county-flow registry (`data/metro/metro_counties.json`,
provisional hand-baked estimates), filtered POI CSVs, boundary polygons, and
network caches are all local artifacts. `welfare_rs/datasource.py` is the one
place that knows where data comes from: `LocalDataSource` (default) reads those
files; `PostgresDataSource` serves identical shapes from the server DB — set
`WELFARE_RS_DATASOURCE=postgres` and `WELFARE_RS_PG_DSN=postgresql://…` (table
schemas are documented in that module).

A metro's first-ever build geocodes boundaries (Nominatim) and downloads the
graph (Overpass) — minutes to tens of minutes per metro. Prefetch everything
once with:

```bash
python scripts/warm_metros.py            # or: python scripts/warm_metros.py miami seattle
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
from welfare_rs import Simulation, build_metro_network, params

net = build_metro_network("miami")                     # cached after first build
sim = Simulation(
    num_agents=80, seed=42, road_network=net,
    persona_csv_path=params.NYC_PERSONA_CSV_PATH,      # behavioural columns only
    disabled_modes=("transit",),
)
day_summaries = sim.run_days(3)
print(sim.summarize())
```

Legacy single-area presets (`build_road_network("nyc_manhattan")`, …) still
work for engine-level experiments; the app itself only surfaces the 8 metros.


## 5. HTTP API (under `/api`)

| Method | Path | Purpose |
|---|---|---|
| GET  | `/api/cities` | the 8 metros, treatments, per-city POI availability, default config |
| POST | `/api/runs` | start (or hit cached) run → `run_id` |
| GET  | `/api/runs/{id}` | run summary + view/bounds/metrics |
| GET  | `/api/runs/{id}/events` | SSE per-day progress |
| GET  | `/api/runs/{id}/timeline?t0&t1&bbox` | windowed, viewport-culled trip/stay keyframes |
| GET  | `/api/runs/{id}/positions?t&bbox` | exact-instant positions (interpolated on real routes) |
| GET  | `/api/runs/{id}/trips/{trip_id}/geometry` | **full** street polyline for one trip (on demand) |
| GET  | `/api/runs/{id}/pois?bbox` | POIs inside the viewport |

Interactive docs at `/docs` (FastAPI/Swagger). `bbox` is `south,west,north,east`.

## 6. POI data

Real POIs come from small per-metro CSVs
(`<cache>/pois/<metro>_leisure_pois.csv`), each produced by one streaming pass
of `data/filter_metro_pois.py` over the raw Placekey/NAICS dump
(`data/poi_children_merged_by_wkt.csv`, ~375 MB, gitignored) clipped to that
metro's core-city polygon.

> **Note:** the raw dump on hand covers the **NYC region only** (despite its
> all-US framing), so today only NYC has real POIs; the other 7 metros fall
> back to synthetic POIs placed on core-city intersections (the "Real POIs"
> toggle disables itself per city). Nationwide POIs arrive with the server DB
> (`metro_pois` table — see `welfare_rs/datasource.py`).

Per the data policy above, none of these files are committed — regenerate them
locally with the filter script, or load them from Postgres on the server.

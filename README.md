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
  commuters** into the core, reduced to the arterial skeleton
  (motorway/trunk/primary/secondary). Dropping residential capillaries outside
  the core is the standard MPO travel-model abstraction and keeps metro graphs
  tractable.

Agent **homes and workplaces come from real LODES8 commutes** (2023, JT01
primary jobs). Each agent is one draw from `metro_od_pairs` — a
`(home census block → work census block)` pair weighted by its job count — with
both block internal points snapped onto the graph. Because the pair is drawn
whole, home and work stay correlated and commute lengths follow the published
distribution rather than two independent uniform draws. The county set that
defines the shell is the same data grouped by home county, so the graph always
covers the homes it needs to place. See `db/07_commute_tables.sql`.

## Try it without any data

A clone has no data: the census inputs are downloaded by scripts, the venue
catalog is a licensed dataset you supply yourself, and the survey population is
IRB-protected and never leaves institutional storage. To see the machinery run
anyway:

```bash
pip install -e .
python data/make_demo_personas.py      # writes FABRICATED personas
python scripts/run_demo.py             # small OSM network, 50 agents, 3 days
```

Those agents are invented and the run reproduces nothing — the engine says so
in its own output every time it loads them. The real pipeline is below.

## Agents come from real people

An agent is not sampled at run time. Populations are built offline and stored,
one row per agent, from three sources used for what each alone can do:

* **LODES** places it — a real home-block → work-block commute, weighted by
  jobs, which also fixes that job's age, earnings and industry band.
* **ACS PUMS**, reweighted to the home tract with IPU, says who lives there:
  one real person's age, own earnings, household income, vehicles, disability
  and household size, *jointly*.
* **The survey** (IRB-FY2026-11354) supplies psychometrics, matched to that
  person on sex, employment, income band and age band.

Non-workers are drawn from **core blocks only**. Their leisure trip departs
from home rather than a workplace, so metro-wide homes would put much of the
population 30–60 km from every venue and swamp every distance-normalised
metric. The population is therefore *people whose day is anchored in the core*.

Full design, approximations and limitations: **`docs/acs-population.md`**.
Build it with `db/fetch_census_data.py` → `db/load_pums.py` → `db/load_acs.py`
→ `db/build_tract_weights.py` → `db/build_population.py`, then verify with
`scripts/verify_acs_population.py`. The repo carries the scripts; the data is
downloaded by them and never committed.

Cars route on **fastest network paths** (per-edge OSM speeds), so freeway
commutes behave like freeway commutes; congestion and weather scale that time.

**Data policy: all data lives on disk (or the lab server's Postgres) — never in
git.** The county-flow registry, filtered POI CSVs, commute pairs, boundary
polygons, and network caches are all local artifacts.
`welfare_rs/datasource.py` is the one place that knows where data comes from:
`LocalDataSource` (default) reads those files; `PostgresDataSource` serves
identical shapes from the server DB — set `WELFARE_RS_DATASOURCE=postgres` and
`WELFARE_RS_PG_DSN=postgresql://…` (table schemas are documented in that
module). Without commute data the engine falls back to its earlier provisional
placement: a county-weighted home plus a uniform workplace inside the core.

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


## 5. Recommenders: one scorer, many configurations

A study is the **No-RS control** plus up to five recommenders the user builds in
the sidebar. Every recommender is the *same* scorer with different knob values —
the presets ("Proximity-led", "Footfall-led", "Viral", …) differ by which weights they
zero out, not by having different machinery:

| Knob | Meaning |
|---|---|
| `w_rating` / `w_reviews` | POI prominence; both move as agents leave feedback |
| `w_relevance` | category fit — near-constant within a subtype, and impersonal by design |
| `w_proximity` | `exp(-d / distance_scale_km)`, straight-line |
| `w_popularity` | live footfall relative to the busiest candidate — the feedback loop |
| `w_personalization` | learned per-agent affinity; the only personal channel |
| `popularity_gamma` | exponent on relative footfall: <1 flattens, >1 concentrates |
| `learning_rate` | how fast feedback moves the user model |
| `welfare_gate` | optional PUP / RM filter on top of the ranking |

Weights are renormalized to sum to 1 and every component is bounded in [0, 1], so
**scores are comparable across configurations**. That matters: the previous
design had `PopularityRecommender` min-max normalizing (top candidate always
~1.0) while `GoogleMapsReplica` emitted a raw weighted sum topping out near 0.75,
so the popularity platform structurally won every cross-platform comparison and
inflated acceptance (eta reads `score - 0.5`) for reasons that were pure scale.

On a cold start the popularity column does not exist yet; its weight is
redistributed over the remaining components rather than counted as a zero, which
would otherwise cap a popularity-heavy recommender at a low score for the whole
warm-up.

### Latent tastes

Each POI carries taste tags recovered from its business name ("Joe's Pizza" →
`pizza`); each agent holds a latent favourite per family (`dining`, `cafe`,
`fitness`, `outdoor`, `culture`, `music`), drawn from the catalog's own supply mix
with a tempering exponent `beta` (`welfare_rs/tastes.py`).

The recommenders **cannot see** the taste. It moves the agent's realised activity
utility, that utility drives like/dislike feedback, and personalization infers the
taste from there — so taste discovery is a genuine learning problem rather than a
lookup. Two caveats worth knowing:

* Only ~50% of real POIs name their type, and *which* ones do is biased (pizza and
  thai self-label; korean and new-american almost never). Unlabelled venues are
  **imputed** a tag from the observed distribution for their family, keyed on a
  hash of the place id, rather than being scored as "definitely not thai" —
  which would have been a systematic penalty on half the catalog.
* Tastes are sampled from name-identifiable venue frequency, which is a proxy for
  cuisine demand, not a measurement of it.

## 6. HTTP API (under `/api`)

| Method | Path | Purpose |
|---|---|---|
| GET  | `/api/cities` | the 8 metros, recommender presets, per-city POI availability, default config |
| POST | `/api/runs` | start (or hit cached) run → `run_id` |
| GET  | `/api/runs/{id}` | run summary + view/bounds/metrics |
| GET  | `/api/runs/{id}/events` | SSE per-day progress |
| GET  | `/api/runs/{id}/timeline?t0&t1&bbox` | windowed, viewport-culled trip/stay keyframes |
| GET  | `/api/runs/{id}/positions?t&bbox` | exact-instant positions (interpolated on real routes) |
| GET  | `/api/runs/{id}/trips/{trip_id}/geometry` | **full** street polyline for one trip (on demand) |
| GET  | `/api/runs/{id}/pois?bbox` | POIs inside the viewport |

Interactive docs at `/docs` (FastAPI/Swagger). `bbox` is `south,west,north,east`.

## 7. POI data

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

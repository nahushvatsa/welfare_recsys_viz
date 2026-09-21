"""FastAPI app for the welfare-oriented activity-travel simulation.

Endpoints (all under ``/api``) let the browser stream only what it needs:

* ``GET  /api/cities``                         — areas, treatments, defaults
* ``POST /api/runs``                           — start (or hit cached) run
* ``GET  /api/runs``                           — list all known runs (no geometry)
* ``GET  /api/runs/{id}``                      — run summary + view/bounds
* ``GET  /api/runs/{id}/events``               — SSE per-day progress
* ``GET  /api/runs/{id}/timeline?condition&seed&t0&t1&bbox`` — windowed keyframes
* ``GET  /api/runs/{id}/positions?condition&seed&t&bbox``    — exact-instant positions
* ``GET  /api/runs/{id}/trips/{trip_id}/geometry?condition&seed`` — full route
* ``GET  /api/runs/{id}/pois?seed&bbox``            — viewport POIs

A "run" is a *study*: several recommender conditions (always including the No-RS
control) swept across several seeds. The geometry endpoints take an optional
``condition`` and ``seed`` to pick which run's trips the map shows; each defaults
to the study's first recommender and first seed. POIs depend on the seed but not
on the condition, so ``/pois`` takes only ``seed``.

The built React frontend (``frontend/dist``) is mounted at ``/`` when present.

Run (dev):
    uvicorn main:app --reload --port 8000     # from the backend/ directory
"""

from __future__ import annotations

import json
import os
import queue
import sys
from typing import Optional

# Make the engine importable even without `pip install -e .` (repo root on path).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

import viz
from models import RunRequest
from service import (CITIES, CONTROL, GOVERNOR, MAX_RECOMMENDERS, MAX_STUDIES,
                     TREATMENTS, RunConfig,
                     manager, precomputed_cities, warmed_cities)
from welfare_rs.recommender_systems import RECOMMENDER_PRESETS

# Interactive docs are opt-in. The OpenAPI schema is a machine-readable manual
# for this API — including the exact ceilings on POST /api/runs, the one
# endpoint that costs real CPU — so a public deployment does not advertise it.
# Set WELFARE_RS_DOCS=1 (dev, or a temporary debugging session) to get it back.
_DOCS = os.environ.get("WELFARE_RS_DOCS", "").strip().lower() in ("1", "true", "yes")

app = FastAPI(
    title="welfare-rs",
    version="0.1.0",
    docs_url="/docs" if _DOCS else None,
    redoc_url="/redoc" if _DOCS else None,
    openapi_url="/openapi.json" if _DOCS else None,
)

# Which *other websites* may drive this API from a visitor's browser. In
# production the SPA is served from the same origin as /api, so it needs no
# grant at all and only the dev-server ports below do anything.
#
# This is not access control: the API stays reachable by curl or any script
# regardless, and CORS is enforced by browsers alone. What it stops is the
# specific trick of a third-party page firing POST /api/runs from every visitor
# it gets — which the previous "*" (plus allow_methods="*") explicitly
# permitted, since the preflight for a JSON POST was answered with yes.
# Override with a comma-separated WELFARE_RS_CORS_ORIGINS.
_DEFAULT_ORIGINS = (
    "https://airecsim.cusp.nyu.edu,http://localhost:5173,http://127.0.0.1:5173"
)
CORS_ORIGINS = [
    o.strip()
    for o in os.environ.get("WELFARE_RS_CORS_ORIGINS", _DEFAULT_ORIGINS).split(",")
    if o.strip()
]

app.add_middleware(GZipMiddleware, minimum_size=500)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["content-type"],
)

_BBOX = tuple[float, float, float, float]


def _parse_bbox(bbox: Optional[str]) -> Optional[_BBOX]:
    """Parse ``"south,west,north,east"`` -> tuple, or None."""
    if not bbox:
        return None
    try:
        south, west, north, east = (float(x) for x in bbox.split(","))
        return (south, west, north, east)
    except (ValueError, TypeError):
        raise HTTPException(status_code=400, detail="bbox must be 'south,west,north,east'")


def _require_run(run_id: str):
    """Resolve a run, or raise a status the client can act on.

    The distinction matters for the run list: a study can legitimately be 'done'
    and yet no longer be openable, because only the last few results are kept in
    memory. 410 tells the UI to show it as expired rather than as a broken link.
    """
    run = manager.get_run(run_id)
    if run is not None:
        return run
    job = manager.get_job(run_id)
    if job is not None and job.status == "running":
        raise HTTPException(status_code=202, detail="run still computing")
    if job is not None and job.status == "error":
        raise HTTPException(status_code=500, detail=job.error or "run failed")
    if job is not None and job.status == "cancelled":
        raise HTTPException(status_code=410, detail="run was cancelled")
    if job is not None and job.status == "done":
        raise HTTPException(
            status_code=410,
            detail="this run's results are no longer held in memory (newer runs replaced it)",
        )
    raise HTTPException(status_code=404, detail="unknown run id")


# ── Metadata ─────────────────────────────────────────────────────────────────

@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/resources")
def resources() -> dict:
    """How the box is currently split between running studies.

    Studies share one process pool and each gets an equal share of its slots,
    so this is the only place the split is visible: how many are running, how
    many slots each is entitled to, how many are actually busy, and how much
    RAM headroom is left before the governor throttles new tasks.
    """
    return {"max_studies": MAX_STUDIES, **GOVERNOR.snapshot()}


@app.get("/api/cities")
def cities() -> dict:
    return {
        "cities": [{"key": k, "label": label} for k, label in CITIES],
        # Starting points for the recommender builder. Each is a set of values
        # for the SAME knobs — a "platform" differs by which weights it zeroes,
        # not by having different machinery — so the UI can seed its sliders
        # from one and let the user take it anywhere from there.
        "presets": [
            {"name": name, **values} for name, values in RECOMMENDER_PRESETS.items()
        ],
        "control": CONTROL,
        "max_recommenders": MAX_RECOMMENDERS,
        # Deprecated fixed vocabulary, still accepted on POST /api/runs.
        "treatments": TREATMENTS,
        "pois_available": manager.pois_available(),  # per-city: {key: bool}
        # Advisory: every city above can be selected, but an unwarmed one
        # downloads its network on demand instead of loading it from disk.
        "warmed": warmed_cities(),                   # per-city: {key: bool}
        # Per-city: routing matrices precomputed. A warm-but-not-precomputed
        # metro still runs; it just pays for Dijkstra trees on every run.
        "precomputed": precomputed_cities(),
        # `treatment` is excluded: it is a deprecated inbound alias, and seeding
        # the UI's config object with it would put a dead key in every POST.
        "defaults": RunRequest().model_dump(exclude={"treatment"}),
    }


# ── Runs ─────────────────────────────────────────────────────────────────────

@app.post("/api/runs")
def create_run(req: RunRequest) -> dict:
    run_id, status = manager.submit(req.to_config())
    body = {"run_id": run_id, "status": status}
    if status == "ready":
        run = manager.get_run(run_id)
        if run is not None:
            body["meta"] = run.meta()
    return body


@app.get("/api/runs")
def list_runs() -> dict:
    """Every run this process knows about — so a browser that did not start a
    study (a second tab, a reload, a shared link) can still find and open it."""
    return {"runs": manager.list_runs()}


@app.get("/api/runs/{run_id}")
def get_run(run_id: str) -> dict:
    return _require_run(run_id).meta()


@app.post("/api/runs/{run_id}/cancel")
def cancel_run(run_id: str) -> dict:
    """Cleanly stop a running study; it aborts (no result) within ~one day."""
    return {"run_id": run_id, "cancelled": manager.cancel(run_id)}


@app.get("/api/runs/{run_id}/events")
def run_events(run_id: str) -> StreamingResponse:
    """Server-sent events: per-day progress, then a terminal done/error event."""
    job = manager.get_job(run_id)
    if job is None:
        # Maybe it's already cached (done with no live job).
        if manager.get_run(run_id) is not None:
            def _done():
                yield f"data: {json.dumps({'type': 'done', 'run_id': run_id})}\n\n"
            return StreamingResponse(_done(), media_type="text/event-stream")
        raise HTTPException(status_code=404, detail="unknown run id")

    def _stream():
        q = job.subscribe()
        while True:
            try:
                event = q.get(timeout=15)
            except queue.Empty:
                yield ": keep-alive\n\n"
                continue
            yield f"data: {json.dumps(event)}\n\n"
            if event.get("type") in ("done", "error", "cancelled"):
                break

    return StreamingResponse(
        _stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Geometry / positions (the streamed, viewport-culled payloads) ─────────────

def _require_seed_viz(run_id: str, condition: Optional[str], seed: Optional[int]):
    """Geometry for one (condition, seed) of a study. Unknown or omitted values
    fall back to the study's defaults rather than 404ing, so a stale bookmark
    still renders something."""
    run = _require_run(run_id)
    sv = run.viz(condition, seed)
    if sv is None:
        raise HTTPException(status_code=404, detail="no geometry for this study")
    return run, sv


def _resolved_condition(run, condition: Optional[str]) -> str:
    return condition if condition in run.per_condition else run.default_condition()


@app.get("/api/runs/{run_id}/timeline")
def timeline(
    run_id: str,
    condition: Optional[str] = Query(None),
    seed: Optional[int] = Query(None),
    t0: float = Query(0.0),
    t1: Optional[float] = Query(None),
    bbox: Optional[str] = Query(None),
) -> dict:
    run, sv = _require_seed_viz(run_id, condition, seed)
    hi = sv.time_span if t1 is None else t1
    box = _parse_bbox(bbox)
    return {
        "condition": _resolved_condition(run, condition),
        "seed": sv.seed,
        "t0": t0,
        "t1": hi,
        "trips": viz.window_trips(sv.merged, t0, hi, box),
        "stays": viz.window_stays(sv.merged, t0, hi, box),
    }


@app.get("/api/runs/{run_id}/positions")
def positions(
    run_id: str,
    condition: Optional[str] = Query(None),
    seed: Optional[int] = Query(None),
    t: float = Query(0.0),
    bbox: Optional[str] = Query(None),
) -> dict:
    run, sv = _require_seed_viz(run_id, condition, seed)
    return {
        "condition": _resolved_condition(run, condition),
        "seed": sv.seed,
        "t": t,
        "positions": viz.positions_at(sv.merged, t, _parse_bbox(bbox)),
    }


@app.get("/api/runs/{run_id}/trips/{trip_id}/geometry")
def trip_geometry(
    run_id: str,
    trip_id: str,
    condition: Optional[str] = Query(None),
    seed: Optional[int] = Query(None),
) -> dict:
    _run, sv = _require_seed_viz(run_id, condition, seed)
    mv = sv.trip_index.get(trip_id)
    if mv is None:
        raise HTTPException(status_code=404, detail="unknown trip id")
    return viz.trip_geometry(mv)


@app.get("/api/runs/{run_id}/pois")
def pois(
    run_id: str,
    seed: Optional[int] = Query(None),
    bbox: Optional[str] = Query(None),
    limit: int = Query(4000),
) -> dict:
    # The POI catalog is condition-independent, so this one is keyed by seed only.
    run = _require_run(run_id)
    return {"pois": viz.pois_in_bbox(run.pois_for(seed), _parse_bbox(bbox), limit=limit)}


# ── Static frontend (built React app), mounted last so /api wins ──────────────

class _CachedStaticFiles(StaticFiles):
    """StaticFiles with cache headers that match how Vite names its output.

    Starlette sends only ETag and Last-Modified, no ``Cache-Control``. A browser
    facing a response with no ``Cache-Control`` is free to apply *heuristic*
    caching — typically a fraction of the time since Last-Modified — and reuse it
    without revalidating. For the hashed assets that is harmless; for
    ``index.html`` it is not, because index.html is the only thing that names
    which bundle to load. A stale copy keeps pointing at the previous build, so a
    deploy silently does nothing until the user happens to hard-refresh.

    So: assets are content-hashed and safe to keep forever, while any HTML must
    be revalidated on every load. ``no-cache`` does not mean "do not store" — the
    ETag still makes it a cheap 304 when nothing has changed.
    """

    async def get_response(self, path: str, scope):
        response = await super().get_response(path, scope)
        if path.startswith("assets/") and "." in path.rsplit("/", 1)[-1]:
            response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
        else:
            response.headers["Cache-Control"] = "no-cache"
        return response


_FRONTEND_DIST = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend", "dist")
if os.path.isdir(_FRONTEND_DIST):
    app.mount("/", _CachedStaticFiles(directory=_FRONTEND_DIST, html=True), name="frontend")
else:
    @app.get("/")
    def _no_frontend() -> dict:
        return {
            "message": "Frontend not built. Run `npm install && npm run build` in frontend/, "
                       "or use the Vite dev server (npm run dev) which proxies /api here.",
            "api_docs": "/docs",
        }

"""FastAPI app for the welfare-oriented activity-travel simulation.

Endpoints (all under ``/api``) let the browser stream only what it needs:

* ``GET  /api/cities``                         — areas, treatments, defaults
* ``POST /api/runs``                           — start (or hit cached) run
* ``GET  /api/runs/{id}``                      — run summary + view/bounds
* ``GET  /api/runs/{id}/events``               — SSE per-day progress
* ``GET  /api/runs/{id}/timeline?seed&t0&t1&bbox``  — windowed trip/stay keyframes
* ``GET  /api/runs/{id}/positions?seed&t&bbox``     — exact-instant positions
* ``GET  /api/runs/{id}/trips/{trip_id}/geometry?seed`` — full route (on demand)
* ``GET  /api/runs/{id}/pois?seed&bbox``            — viewport POIs

A "run" is a multi-seed *study*; the geometry endpoints take an optional ``seed``
to pick which seed's trips the map shows (defaults to the study's first seed).

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
from service import AREAS, TREATMENTS, RunConfig, manager

app = FastAPI(title="welfare-rs", version="0.1.0")

app.add_middleware(GZipMiddleware, minimum_size=500)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # dev: Vite serves the frontend from another origin
    allow_methods=["*"],
    allow_headers=["*"],
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
    run = manager.get_run(run_id)
    if run is not None:
        return run
    job = manager.get_job(run_id)
    if job is not None and job.status == "running":
        raise HTTPException(status_code=202, detail="run still computing")
    if job is not None and job.status == "error":
        raise HTTPException(status_code=500, detail=job.error or "run failed")
    raise HTTPException(status_code=404, detail="unknown run id")


# ── Metadata ─────────────────────────────────────────────────────────────────

@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/cities")
def cities() -> dict:
    return {
        "areas": [{"key": k, "label": label} for k, label in AREAS],
        "treatments": TREATMENTS,
        "pois_available": manager.pois_available(),
        "defaults": RunRequest().model_dump(),
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

def _require_seed_viz(run_id: str, seed: Optional[int]):
    run = _require_run(run_id)
    sv = run.viz(seed)
    if sv is None:
        raise HTTPException(status_code=404, detail="no geometry for this study")
    return sv


@app.get("/api/runs/{run_id}/timeline")
def timeline(
    run_id: str,
    seed: Optional[int] = Query(None),
    t0: float = Query(0.0),
    t1: Optional[float] = Query(None),
    bbox: Optional[str] = Query(None),
) -> dict:
    sv = _require_seed_viz(run_id, seed)
    hi = sv.time_span if t1 is None else t1
    box = _parse_bbox(bbox)
    return {
        "seed": sv.seed,
        "t0": t0,
        "t1": hi,
        "trips": viz.window_trips(sv.merged, t0, hi, box),
        "stays": viz.window_stays(sv.merged, t0, hi, box),
    }


@app.get("/api/runs/{run_id}/positions")
def positions(
    run_id: str,
    seed: Optional[int] = Query(None),
    t: float = Query(0.0),
    bbox: Optional[str] = Query(None),
) -> dict:
    sv = _require_seed_viz(run_id, seed)
    return {"seed": sv.seed, "t": t, "positions": viz.positions_at(sv.merged, t, _parse_bbox(bbox))}


@app.get("/api/runs/{run_id}/trips/{trip_id}/geometry")
def trip_geometry(run_id: str, trip_id: str, seed: Optional[int] = Query(None)) -> dict:
    sv = _require_seed_viz(run_id, seed)
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
    sv = _require_seed_viz(run_id, seed)
    return {"pois": viz.pois_in_bbox(sv.pois, _parse_bbox(bbox), limit=limit)}


# ── Static frontend (built React app), mounted last so /api wins ──────────────

_FRONTEND_DIST = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend", "dist")
if os.path.isdir(_FRONTEND_DIST):
    app.mount("/", StaticFiles(directory=_FRONTEND_DIST, html=True), name="frontend")
else:
    @app.get("/")
    def _no_frontend() -> dict:
        return {
            "message": "Frontend not built. Run `npm install && npm run build` in frontend/, "
                       "or use the Vite dev server (npm run dev) which proxies /api here.",
            "api_docs": "/docs",
        }

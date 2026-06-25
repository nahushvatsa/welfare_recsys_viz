"""Run management for the FastAPI backend.

Owns the simulation: builds/caches the OSM road network per area, runs the ABM
(capturing per-day timelines), and keeps completed runs in an in-process cache
keyed by config. Sims execute on a *single* worker thread — this serialises them
so the global ``params.SIMPLIFICATION_TOGGLES["CAR_ONLY_MODE"]`` flag (set per
run for the multimodal toggle) can't race across concurrent requests, and it
exposes per-day progress for the SSE endpoint.

This module is the only Streamlit-free successor to the old ``app.py`` engine
glue; it imports the published :mod:`welfare_rs` API plus the backend's
``viz`` helpers.
"""

from __future__ import annotations

import hashlib
import os
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional

from welfare_rs import params
from welfare_rs.geo import build_road_network, haversine_km
from welfare_rs.simulation import Simulation

import viz

# ── Treatments / areas (mirror the old Streamlit UI) ─────────────────────────

TREATMENTS = ["No RS", "Standard RS", "PUP", "RM", "PUP+RM"]
_WELFARE_MODE = {"PUP": "pup", "RM": "rm", "PUP+RM": "pup_rm"}

# (city-preset key, friendly label) — surfaced to the frontend as area choices.
AREAS = [
    ("nyc_manhattan", "Manhattan"),
    ("nyc_lower_manhattan", "Lower Manhattan"),
    ("brooklyn_full", "Brooklyn (full)"),
    ("brooklyn_downtown_park_slope", "Downtown BK / Park Slope"),
    ("brooklyn_south_prospect_bay_ridge", "South BK · Prospect / Bay Ridge"),
]
_AREA_LABELS = dict(AREAS)


# ── Config / run records ─────────────────────────────────────────────────────

@dataclass(frozen=True)
class RunConfig:
    city: str = "nyc_manhattan"
    num_agents: int = 80
    num_days: int = 3
    seed: int = 42
    treatment: str = "Standard RS"
    multimodal: bool = False
    use_real_pois: bool = True
    pup_alpha: float = 0.6
    rm_epsilon: float = 0.3

    def run_id(self) -> str:
        key = (
            self.city, int(self.num_agents), int(self.num_days), int(self.seed),
            self.treatment, bool(self.multimodal), bool(self.use_real_pois),
            round(float(self.pup_alpha), 4), round(float(self.rm_epsilon), 4),
        )
        return hashlib.sha1(repr(key).encode()).hexdigest()[:16]


@dataclass
class Run:
    run_id: str
    config: RunConfig
    merged: Dict[int, dict]
    trip_index: Dict[str, dict]
    pois: List[dict]
    summary: dict
    day_summaries: List[dict]
    view: dict
    bounds: dict
    num_days: int
    num_intersections: int
    poi_count: int
    poi_source: str
    time_span: int

    def meta(self) -> dict:
        """Lightweight payload returned on run creation / status (no geometry)."""
        return {
            "run_id": self.run_id,
            "config": asdict(self.config),
            "summary": self.summary,
            "day_summaries": self.day_summaries,
            "view": self.view,
            "bounds": self.bounds,
            "num_days": self.num_days,
            "num_intersections": self.num_intersections,
            "poi_count": self.poi_count,
            "poi_source": self.poi_source,
            "time_span": self.time_span,
            "area_label": _AREA_LABELS.get(self.config.city, self.config.city),
        }


# ── Job (background sim execution + progress) ────────────────────────────────

@dataclass
class Job:
    run_id: str
    status: str = "running"  # running | done | error
    current: int = 0
    total: int = 0
    error: Optional[str] = None
    _subs: List["queue.Queue"] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def publish(self, event: dict) -> None:
        with self._lock:
            for q in self._subs:
                q.put(event)

    def subscribe(self) -> "queue.Queue":
        q: "queue.Queue" = queue.Queue()
        with self._lock:
            self._subs.append(q)
            # Replay current state so a late subscriber isn't stuck waiting.
            if self.status == "running":
                q.put({"type": "progress", "current": self.current, "total": self.total})
            elif self.status == "done":
                q.put({"type": "done", "run_id": self.run_id})
            elif self.status == "error":
                q.put({"type": "error", "message": self.error or "run failed"})
        return q


# ── Recommender wiring (welfare gate) ────────────────────────────────────────

def _make_recommender_factory(treatment: str, pup_alpha: float, rm_epsilon: float):
    """Return a recommender_factory wiring the welfare gate, or None for the
    pass-through Standard RS / No RS conditions."""
    if treatment not in _WELFARE_MODE:
        return None
    from welfare_rs import TravelCostEstimator, WelfareAwareOrchestrator

    mode = _WELFARE_MODE[treatment]

    def factory(_sim, base_stack):
        return WelfareAwareOrchestrator(
            base_stack,
            TravelCostEstimator(),
            mode=mode,
            pup_alpha=pup_alpha,
            rm_epsilon=rm_epsilon,
            coord_scale_km=1.0,
            coord_distance_km=haversine_km,
        )

    return factory


def _resolve_poi_path() -> Optional[str]:
    """Locate the filtered NYC POI CSV, robust to a remapped HOME env var."""
    candidates = [params.NYC_POI_CSV_PATH]
    try:
        import pwd

        real_home = pwd.getpwuid(os.getuid()).pw_dir
        candidates.append(os.path.join(real_home, ".cache", "welfare_rs", "pois", "nyc_leisure_pois.csv"))
    except Exception:
        pass
    for path in candidates:
        if path and os.path.exists(path):
            return path
    return None


# ── Run manager ──────────────────────────────────────────────────────────────

class RunManager:
    """Builds and caches networks + runs; serialises sim execution."""

    def __init__(self, max_runs: int = 16):
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sim")
        self._networks: Dict[str, object] = {}
        self._net_lock = threading.Lock()
        self._runs: Dict[str, Run] = {}
        self._jobs: Dict[str, Job] = {}
        self._lock = threading.Lock()
        self._max_runs = max_runs

    # -- networks --
    def get_network(self, city: str):
        with self._net_lock:
            net = self._networks.get(city)
            if net is None:
                net = build_road_network(city)
                self._networks[city] = net
            return net

    def pois_available(self) -> bool:
        return _resolve_poi_path() is not None

    # -- runs / jobs --
    def get_run(self, run_id: str) -> Optional[Run]:
        return self._runs.get(run_id)

    def get_job(self, run_id: str) -> Optional[Job]:
        return self._jobs.get(run_id)

    def submit(self, config: RunConfig) -> tuple[str, str]:
        """Return ``(run_id, status)``. status: ready | running."""
        run_id = config.run_id()
        with self._lock:
            if run_id in self._runs:
                return run_id, "ready"
            if run_id in self._jobs and self._jobs[run_id].status == "running":
                return run_id, "running"
            job = Job(run_id=run_id, total=int(config.num_days))
            self._jobs[run_id] = job
        self._executor.submit(self._execute, run_id, config)
        return run_id, "running"

    def _execute(self, run_id: str, config: RunConfig) -> None:
        job = self._jobs[run_id]
        try:
            run = self._build_run(run_id, config, job)
            with self._lock:
                self._runs[run_id] = run
                # Bound the cache (FIFO).
                while len(self._runs) > self._max_runs:
                    oldest = next(iter(self._runs))
                    self._runs.pop(oldest)
            job.status = "done"
            job.publish({"type": "done", "run_id": run_id})
        except Exception as exc:  # surface to the SSE stream
            job.status = "error"
            job.error = f"{type(exc).__name__}: {exc}"
            job.publish({"type": "error", "message": job.error})

    def _build_run(self, run_id: str, cfg: RunConfig, job: Job) -> Run:
        network = self.get_network(cfg.city)

        # Mode model: paper default is car-only; the toggle re-enables the
        # multimodal model. Safe to set globally here because sims are serialised.
        params.SIMPLIFICATION_TOGGLES["CAR_ONLY_MODE"] = not cfg.multimodal

        use_real_pois = bool(cfg.use_real_pois) and self.pois_available()
        poi_csv_path = _resolve_poi_path() if use_real_pois else None

        sim = Simulation(
            num_agents=int(cfg.num_agents),
            seed=int(cfg.seed),
            use_recommenders=(cfg.treatment != "No RS"),
            persona_csv_path=params.NYC_PERSONA_CSV_PATH,
            road_network=network,
            poi_csv_path=poi_csv_path,
            disabled_modes=("transit",),
            recommender_factory=_make_recommender_factory(cfg.treatment, cfg.pup_alpha, cfg.rm_epsilon),
        )

        per_day: List[Dict[int, dict]] = []

        def _capture(day_idx, s):
            # Fires while this day's trips are still on the agents.
            per_day.append(viz.build_timelines(s))

        def _progress(day, total):
            job.current, job.total = day, total
            job.publish({"type": "progress", "current": day, "total": total})

        day_summaries = sim.run_days(int(cfg.num_days), progress=_progress, on_day_complete=_capture)

        merged = viz.merge_day_timelines(per_day)
        trip_index = viz.build_trip_index(merged)

        net = sim.road_network
        south, west, north, east = net.bounds
        pois = [
            {"lon": round(p.location[1], 6), "lat": round(p.location[0], 6),
             "category": p.category, "label": f"{p.name} · {p.category}"}
            for p in sim.place_catalog
        ]

        summary = sim.summarize()
        leisure_utils = [t.utility for a in sim.agents for t in a.trips if t.purpose == "leisure"]
        summary["mean_leisure_net_utility"] = (
            sum(leisure_utils) / len(leisure_utils) if leisure_utils else 0.0
        )
        summary["poi_count"] = len(sim.place_catalog)

        return Run(
            run_id=run_id,
            config=cfg,
            merged=merged,
            trip_index=trip_index,
            pois=pois,
            summary=summary,
            day_summaries=day_summaries,
            view={"latitude": net.center[0], "longitude": net.center[1]},
            bounds={"south": south, "west": west, "north": north, "east": east},
            num_days=int(cfg.num_days),
            num_intersections=net.num_base_nodes,
            poi_count=len(sim.place_catalog),
            poi_source="real NYC dataset" if use_real_pois else "synthetic",
            time_span=viz.time_span(merged),
        )


# Module-level singleton used by the FastAPI app.
manager = RunManager()

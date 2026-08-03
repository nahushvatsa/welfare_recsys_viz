"""Run management for the FastAPI backend (multi-seed studies).

A *study* is one treatment swept across N random seeds. For each seed we run the
selected treatment **and** a matched No-RS counterfactual (same seed → same agent
population), which is what the paper's Table 2 (over-recommendation cost) requires.
Seeds are embarrassingly parallel, so they fan out across a ``ProcessPoolExecutor``:

* Threads wouldn't help — the agent loop is pure-Python (GIL-bound), and the
  per-run ``params.SIMPLIFICATION_TOGGLES["CAR_ONLY_MODE"]`` global would race.
  Separate processes each get their own module globals, so the toggle is set
  safely per worker and there is no cross-seed contention.
* The OSM network is loaded from the on-disk GraphML cache (no Overpass at run
  time); the parent pre-builds it once so workers never race to download it. A
  worker keeps its (POI-augmented) network in a process-global cache and reuses it
  across the seeds it handles — the geometry is seed-independent.

Metrics are computed the paper's way (welfare_rs.experiment_harness): Table 1 over
the **last (evaluation) day's** leisure trips per run, then aggregated across seeds
(σ_U = std of per-seed Ū); Table 2 paired against the No-RS run per seed.
"""

from __future__ import annotations

import hashlib
import os
import sys

# Make ``viz`` and ``welfare_rs`` importable even in spawned worker processes,
# which re-import this module fresh (macOS uses the 'spawn' start method).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import multiprocessing as mp
import queue
import threading
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import numpy as np

from welfare_rs import params
from welfare_rs.datasource import get_datasource
from welfare_rs.experiment_harness import table1_metrics, table2_metrics
from welfare_rs.geo import haversine_km
from welfare_rs.metro import build_network
from welfare_rs.simulation import Simulation, SimulationCancelled

import viz

# ── Treatments / cities ──────────────────────────────────────────────────────

TREATMENTS = ["No RS", "Standard RS", "PUP", "RM", "PUP+RM"]
_WELFARE_MODE = {"PUP": "pup", "RM": "rm", "PUP+RM": "pup_rm"}

# (metro key, friendly label) — the two-layer metros surfaced to the frontend.
#
# ALL metros are offered, warmed or not. Picking an unwarmed one makes the run
# build its network on demand, which goes to Overpass from inside the web
# worker — slow at best, and while this host is blocked by overpass-api.de it
# hangs on a 180 s connect timeout per sub-query before failing. That is a
# deliberate development-time choice: the operator knows which metros are warm.
# ``warmed`` in the /api/cities payload says which are ready without having to
# guess. Warming itself stays an offline step (scripts/warm_metros.py).
CITIES = [(key, spec["label"]) for key, spec in params.METRO_PARAMS["metros"].items()]
_CITY_LABELS = dict(CITIES)
DEFAULT_CITY = params.METRO_PARAMS["default_metro"]


def warmed_cities() -> dict:
    """``{metro key: warm pickle exists}`` — advisory, read fresh each call so
    a metro warmed while the service is up shows up without a restart."""
    cache_dir = params.GEO_PARAMS["cache_dir"]
    return {
        key: os.path.exists(os.path.join(cache_dir, "warmed", f"{key}_metro.pkl"))
        for key, _ in CITIES
    }

MAX_SEEDS = 12


# ── Config / run records ─────────────────────────────────────────────────────

@dataclass(frozen=True)
class RunConfig:
    city: str = DEFAULT_CITY
    num_agents: int = 80
    num_days: int = 3
    seed: int = 42          # base seed; the study sweeps seed .. seed+num_seeds-1
    num_seeds: int = 3
    treatment: str = "Standard RS"
    multimodal: bool = False
    use_real_pois: bool = True
    pup_alpha: float = 0.6
    rm_epsilon: float = 0.3

    def seeds(self) -> List[int]:
        n = max(1, min(int(self.num_seeds), MAX_SEEDS))
        return [int(self.seed) + i for i in range(n)]

    def run_id(self) -> str:
        key = (
            self.city, int(self.num_agents), int(self.num_days), int(self.seed),
            int(self.num_seeds), self.treatment, bool(self.multimodal),
            bool(self.use_real_pois), round(float(self.pup_alpha), 4),
            round(float(self.rm_epsilon), 4),
        )
        return hashlib.sha1(repr(key).encode()).hexdigest()[:16]


@dataclass
class SeedViz:
    """Per-seed geometry for the map (the *treatment* condition's trips)."""
    seed: int
    merged: Dict[int, dict]
    trip_index: Dict[str, dict]
    pois: List[dict]
    time_span: int


@dataclass
class Run:
    run_id: str
    config: RunConfig
    seeds: List[int]
    per_seed: Dict[int, SeedViz]
    table1: List[dict]            # rows: No RS, then the treatment (if not No RS)
    table2: Optional[dict]        # treatment's ORC row, or None for the No-RS study
    headline: dict
    aggregate: dict
    view: dict
    bounds: dict                  # full metro network extent
    core_bounds: dict             # principal-city extent (initial map frame)
    num_days: int
    num_intersections: int
    poi_count: int
    poi_source: str

    def meta(self) -> dict:
        """Aggregated metrics + per-seed index (no geometry)."""
        return {
            "run_id": self.run_id,
            "config": asdict(self.config),
            "seeds": self.seeds,
            "default_seed": self.seeds[0] if self.seeds else self.config.seed,
            "time_spans": {str(s): v.time_span for s, v in self.per_seed.items()},
            "table1": self.table1,
            "table2": self.table2,
            "headline": self.headline,
            "aggregate": self.aggregate,
            "view": self.view,
            "bounds": self.bounds,
            "core_bounds": self.core_bounds,
            "num_days": self.num_days,
            "num_intersections": self.num_intersections,
            "poi_count": self.poi_count,
            "poi_source": self.poi_source,
            "city_label": _CITY_LABELS.get(self.config.city, self.config.city),
        }

    def viz(self, seed: Optional[int]) -> Optional[SeedViz]:
        if seed is not None and int(seed) in self.per_seed:
            return self.per_seed[int(seed)]
        return self.per_seed.get(self.seeds[0]) if self.seeds else None


# ── Job (background study execution + progress) ──────────────────────────────

@dataclass
class Job:
    run_id: str
    status: str = "running"  # running | done | error | cancelled
    current: int = 0
    total: int = 0
    error: Optional[str] = None
    # Cancellation: `cancel` is the in-process signal (inline single-seed path);
    # `mp_cancel` is a multiprocessing Event shared with spawned seed workers,
    # created per study while the process pool is live. Both are checked between
    # days by the engine's should_stop, so a stop takes effect within one day.
    cancel: threading.Event = field(default_factory=threading.Event)
    mp_cancel: Optional[object] = None
    _subs: List["queue.Queue"] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def request_cancel(self) -> None:
        self.cancel.set()
        if self.mp_cancel is not None:
            try:
                self.mp_cancel.set()
            except Exception:
                pass

    def publish(self, event: dict) -> None:
        with self._lock:
            for q in self._subs:
                q.put(event)

    def subscribe(self) -> "queue.Queue":
        q: "queue.Queue" = queue.Queue()
        with self._lock:
            self._subs.append(q)
            if self.status == "running":
                q.put({"type": "progress", "current": self.current, "total": self.total})
            elif self.status == "done":
                q.put({"type": "done", "run_id": self.run_id})
            elif self.status == "error":
                q.put({"type": "error", "message": self.error or "run failed"})
            elif self.status == "cancelled":
                q.put({"type": "cancelled", "run_id": self.run_id})
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


def pois_available() -> Dict[str, bool]:
    """Per-city real-POI availability (drives the frontend toggle)."""
    ds = get_datasource()
    out = {}
    for key, _label in CITIES:
        try:
            out[key] = ds.has_pois(key)
        except Exception:
            out[key] = False
    return out


# ── Worker: run one seed (treatment + matched No-RS) ─────────────────────────
#
# Module-level so it is picklable by ProcessPoolExecutor. Each worker keeps the
# (POI-augmented) network for a (city, use_real_pois) in a process-global cache
# and reuses it across the seeds it handles; the geometry is seed-independent.

_WORKER_NET_CACHE: Dict[tuple, object] = {}


def _worker_network(city: str, use_real_pois: bool):
    key = (city, bool(use_real_pois))
    net = _WORKER_NET_CACHE.get(key)
    if net is None:
        net = build_network(city)  # metro two-layer or legacy preset
        _WORKER_NET_CACHE[key] = net
    return net


def _build_sim(cfg: dict, seed: int, treatment: str, network, poi_rows):
    ds = get_datasource()
    return Simulation(
        num_agents=int(cfg["num_agents"]),
        seed=int(seed),
        use_recommenders=(treatment != "No RS"),
        persona_csv_path=ds.persona_csv_path() or params.NYC_PERSONA_CSV_PATH,
        road_network=network,
        poi_rows=poi_rows,
        disabled_modes=("transit",),
        recommender_factory=_make_recommender_factory(
            treatment, cfg["pup_alpha"], cfg["rm_epsilon"]
        ),
    )


def _run_one_seed(cfg: dict, seed: int, cancel_event=None) -> dict:
    """Run the treatment (+ matched No-RS counterfactual) for one seed.

    Returns a picklable dict of per-seed metrics and the treatment viz payload.
    Raises SimulationCancelled if ``cancel_event`` is set between days.
    """
    # Mode model: paper default is car-only; the toggle re-enables multimodal.
    # Safe to set on the module global — this is the worker's own process.
    params.SIMPLIFICATION_TOGGLES["CAR_ONLY_MODE"] = not bool(cfg["multimodal"])

    treatment = cfg["treatment"]
    poi_rows = get_datasource().poi_rows(cfg["city"]) if bool(cfg["use_real_pois"]) else None
    use_real_pois = bool(poi_rows)
    network = _worker_network(cfg["city"], use_real_pois)

    # Cooperative cancellation: checked between days by the engine. `cancel_event`
    # is the Job's threading.Event (inline) or a shared mp Event (pool workers).
    should_stop = (lambda: cancel_event.is_set()) if cancel_event is not None else None

    # Treatment run — capture per-day timelines for the map.
    per_day: List[Dict[int, dict]] = []
    treat_sim = _build_sim(cfg, seed, treatment, network, poi_rows)
    treat_sim.run_days(
        int(cfg["num_days"]),
        on_day_complete=lambda _d, s: per_day.append(viz.build_timelines(s)),
        should_stop=should_stop,
    )

    # Matched No-RS counterfactual (metrics only — no viz needed). For the No-RS
    # study itself, the treatment run *is* the baseline; reuse it.
    if treatment == "No RS":
        no_rs_sim = treat_sim
    else:
        no_rs_sim = _build_sim(cfg, seed, "No RS", network, poi_rows)
        no_rs_sim.run_days(int(cfg["num_days"]), should_stop=should_stop)

    t1_treatment = table1_metrics(treat_sim)
    t1_no_rs = t1_treatment if treatment == "No RS" else table1_metrics(no_rs_sim)
    t2 = None if treatment == "No RS" else table2_metrics(treat_sim, no_rs_sim)

    merged = viz.merge_day_timelines(per_day)
    net = treat_sim.road_network
    south, west, north, east = net.bounds
    csouth, cwest, cnorth, ceast = net.core_bounds
    pois = [
        {"lon": round(p.location[1], 6), "lat": round(p.location[0], 6),
         "category": p.category, "label": f"{p.name} · {p.category}"}
        for p in treat_sim.place_catalog
    ]
    summ = treat_sim.summarize()

    return {
        "seed": int(seed),
        "t1_treatment": t1_treatment,
        "t1_no_rs": t1_no_rs,
        "t2": t2,
        "summary": {
            "recommendation_acceptance_rate": summ.get("recommendation_acceptance_rate", 0.0),
            "avg_travel_time_min": summ.get("avg_travel_time_min", 0.0),
        },
        "viz": {
            "merged": merged,
            "trip_index": viz.build_trip_index(merged),
            "pois": pois,
            "time_span": viz.time_span(merged),
        },
        "net": {
            # Initial map frame: the principal city; agents commute in from
            # the wider metro bounds visible on zoom-out.
            "view": {"latitude": net.core_center[0], "longitude": net.core_center[1]},
            "bounds": {"south": south, "west": west, "north": north, "east": east},
            "core_bounds": {"south": csouth, "west": cwest, "north": cnorth, "east": ceast},
            "num_intersections": net.num_base_nodes,
            "poi_count": len(treat_sim.place_catalog),
            "poi_source": "real dataset" if use_real_pois else "synthetic",
        },
    }


# ── Aggregation across seeds ─────────────────────────────────────────────────

def _agg_table1_row(condition: str, per_seed_rows: List[dict]) -> dict:
    """Average a Table-1 condition across seeds; σ_U = std of per-seed Ū."""
    means = [r["mean_utility"] for r in per_seed_rows]
    return {
        "condition": condition,
        "mean_utility": float(np.mean(means)),
        "sigma_u": float(np.std(means)) if len(means) > 1 else 0.0,
        "neg_rate": float(np.mean([r["neg_rate"] for r in per_seed_rows])),
        "abstention_rate": float(np.mean([r["abstention_rate"] for r in per_seed_rows])),
        "gini": float(np.mean([r["gini"] for r in per_seed_rows])),
        "n_leisure_trips": int(np.sum([r["n_leisure_trips"] for r in per_seed_rows])),
    }


def _agg_table2(treatment: str, per_seed_t2: List[dict]) -> dict:
    return {
        "condition": treatment,
        "harmed_pct": float(np.mean([t["harmed_pct"] for t in per_seed_t2])),
        "improved_pct": float(np.mean([t["improved_pct"] for t in per_seed_t2])),
        "mean_orc": float(np.mean([t["mean_orc"] for t in per_seed_t2])),
        "n_matched": int(np.sum([t["n_matched"] for t in per_seed_t2])),
    }


# ── Run manager ──────────────────────────────────────────────────────────────

class RunManager:
    """Builds and caches studies; serialises study execution (one study at a
    time), parallelising seeds within a study across processes."""

    def __init__(self, max_runs: int = 12):
        # One study at a time keeps memory bounded and avoids oversubscribing the
        # CPU (seeds already fan out across all cores inside a study).
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="study")
        self._runs: Dict[str, Run] = {}
        self._jobs: Dict[str, Job] = {}
        self._lock = threading.Lock()
        self._max_runs = max_runs

    def pois_available(self) -> bool:
        return pois_available()

    def get_run(self, run_id: str) -> Optional[Run]:
        return self._runs.get(run_id)

    def get_job(self, run_id: str) -> Optional[Job]:
        return self._jobs.get(run_id)

    def cancel(self, run_id: str) -> bool:
        """Signal a running study to stop. Returns True if a running job was
        signalled. The study aborts (no result stored) within ~one day."""
        job = self._jobs.get(run_id)
        if job is not None and job.status == "running":
            job.request_cancel()
            return True
        return False

    def submit(self, config: RunConfig) -> tuple[str, str]:
        """Return ``(run_id, status)``. status: ready | running."""
        run_id = config.run_id()
        with self._lock:
            if run_id in self._runs:
                return run_id, "ready"
            if run_id in self._jobs and self._jobs[run_id].status == "running":
                return run_id, "running"
            job = Job(run_id=run_id, total=len(config.seeds()))
            self._jobs[run_id] = job
        self._executor.submit(self._execute, run_id, config)
        return run_id, "running"

    def _execute(self, run_id: str, config: RunConfig) -> None:
        job = self._jobs[run_id]
        try:
            run = self._build_study(run_id, config, job)
            with self._lock:
                self._runs[run_id] = run
                while len(self._runs) > self._max_runs:
                    oldest = next(iter(self._runs))
                    self._runs.pop(oldest)
            job.status = "done"
            job.publish({"type": "done", "run_id": run_id})
        except SimulationCancelled:
            job.status = "cancelled"
            job.publish({"type": "cancelled", "run_id": run_id})
        except Exception as exc:
            job.status = "error"
            job.error = f"{type(exc).__name__}: {exc}"
            job.publish({"type": "error", "message": job.error})

    def _build_study(self, run_id: str, cfg: RunConfig, job: Job) -> Run:
        seeds = cfg.seeds()
        job.total = len(seeds)

        # Pre-build the network once so worker processes only ever *load* the
        # disk caches (no concurrent Overpass download / warm-pickle race).
        build_network(cfg.city)

        cfg_dict = asdict(cfg)
        results: Dict[int, dict] = {}

        def _collect(res: dict) -> None:
            results[res["seed"]] = res
            job.current = len(results)
            job.publish({"type": "progress", "current": job.current, "total": job.total})

        if len(seeds) == 1:
            # Inline: the Job's threading.Event is visible to should_stop directly.
            _collect(_run_one_seed(cfg_dict, seeds[0], cancel_event=job.cancel))
        else:
            workers = min(len(seeds), os.cpu_count() or 1)
            # 'spawn' (macOS default): fork-after-threads under uvicorn is unsafe.
            ctx = mp.get_context("spawn")
            # A Manager Event is shareable with spawned workers; setting it makes
            # each worker's should_stop fire at its next day boundary.
            manager_proc = ctx.Manager()
            job.mp_cancel = manager_proc.Event()
            if job.cancel.is_set():  # cancel arrived during setup
                job.mp_cancel.set()
            try:
                with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
                    futs = {pool.submit(_run_one_seed, cfg_dict, s, job.mp_cancel): s for s in seeds}
                    try:
                        for fut in as_completed(futs):
                            _collect(fut.result())
                    except SimulationCancelled:
                        for f in futs:
                            f.cancel()  # drop seeds not yet started
                        raise
            finally:
                job.mp_cancel = None
                manager_proc.shutdown()

        return self._assemble(run_id, cfg, seeds, results)

    def _assemble(self, run_id: str, cfg: RunConfig, seeds: List[int],
                  results: Dict[int, dict]) -> Run:
        ordered = [results[s] for s in seeds if s in results]

        # Table 1: No RS row always; the treatment row when it isn't No RS.
        table1 = [_agg_table1_row("No RS", [r["t1_no_rs"] for r in ordered])]
        if cfg.treatment != "No RS":
            table1.append(_agg_table1_row(cfg.treatment, [r["t1_treatment"] for r in ordered]))

        table2 = None
        if cfg.treatment != "No RS":
            table2 = _agg_table2(cfg.treatment, [r["t2"] for r in ordered])

        treat_row = table1[-1]  # treatment row, or No RS for the baseline study
        headline = {
            "mean_utility": treat_row["mean_utility"],
            "sigma_u": treat_row["sigma_u"],
            "neg_rate": treat_row["neg_rate"],
            "harmed_pct": table2["harmed_pct"] if table2 else None,
            "improved_pct": table2["improved_pct"] if table2 else None,
            "mean_orc": table2["mean_orc"] if table2 else None,
        }
        aggregate = {
            "acceptance_rate": float(np.mean(
                [r["summary"]["recommendation_acceptance_rate"] for r in ordered])),
            "avg_travel_time_min": float(np.mean(
                [r["summary"]["avg_travel_time_min"] for r in ordered])),
        }

        per_seed = {
            r["seed"]: SeedViz(
                seed=r["seed"],
                merged={int(k): v for k, v in r["viz"]["merged"].items()},
                trip_index=r["viz"]["trip_index"],
                pois=r["viz"]["pois"],
                time_span=r["viz"]["time_span"],
            )
            for r in ordered
        }
        net0 = ordered[0]["net"]

        return Run(
            run_id=run_id,
            config=cfg,
            seeds=seeds,
            per_seed=per_seed,
            table1=table1,
            table2=table2,
            headline=headline,
            aggregate=aggregate,
            view=net0["view"],
            bounds=net0["bounds"],
            core_bounds=net0["core_bounds"],
            num_days=int(cfg.num_days),
            num_intersections=net0["num_intersections"],
            poi_count=net0["poi_count"],
            poi_source=net0["poi_source"],
        )


# Module-level singleton used by the FastAPI app.
manager = RunManager()

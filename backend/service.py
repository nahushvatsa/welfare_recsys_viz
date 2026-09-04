"""Run management for the FastAPI backend (multi-condition, multi-seed studies).

A *study* is a set of recommender **conditions** swept across N random seeds. The
No-RS control is always one of them — it is the baseline every Table-2 row is
paired against, and its movement is worth watching on the map in its own right.

The unit of parallelism is one simulation: every ``(seed, condition)`` pair is a
separate task on a ``ProcessPoolExecutor``. A 3-seed study over 4 recommenders is
15 simulations running at once, not 3 sequential pairs. Why processes, and what
they do and don't share:

* Threads wouldn't help — the agent loop is pure-Python (GIL-bound), and the
  per-run ``params.SIMPLIFICATION_TOGGLES["CAR_ONLY_MODE"]`` global would race.
  Separate processes each get their own module globals, so the toggle is set
  safely per worker and there is no cross-task contention.
* The metro graph is *not* shared: the pool uses 'spawn', so each worker
  unpickles its own copy (tens of MB on disk, a few hundred resident). The parent
  pre-builds it once so no worker downloads from Overpass or races to write the
  warm pickle, and after the first read the file is served from page cache. A
  worker caches the network process-globally and reuses it across its tasks.
* The routing matrices (welfare_rs.routing_matrix, 3-35 GB per metro) ARE shared:
  they are opened ``mmap_mode="r"``, so every worker reads one page-cache copy.
  That is what makes a wide fan-out affordable.

Table 2 pairs a recommender against No RS agent-by-agent, which would otherwise
force both runs into one process. Instead each task returns its per-agent eval-day
utilities and the pairing happens here, in the parent (see ``_assemble``).

Metrics are computed the paper's way (welfare_rs.experiment_harness): Table 1 over
the **last (evaluation) day's** leisure trips per run, then aggregated across seeds
(σ_U = std of per-seed Ū); Table 2 paired against the No-RS run per seed.
"""

from __future__ import annotations

import hashlib
import math
import os
import sys

# Make ``viz`` and ``welfare_rs`` importable even in spawned worker processes,
# which re-import this module fresh (macOS uses the 'spawn' start method).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import multiprocessing as mp
import queue
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from welfare_rs import metrics as sim_metrics
from welfare_rs import params, routing_matrix
from welfare_rs.datasource import get_datasource
from welfare_rs.experiment_harness import (
    eval_day_leisure_utilities,
    table1_metrics,
    table2_from_utilities,
)
from welfare_rs.geo import haversine_km
from welfare_rs.metro import build_network
from welfare_rs.recommender_systems import RECOMMENDER_PRESETS, RecommenderConfig
from welfare_rs.simulation import Simulation, SimulationCancelled

import viz

# ── Recommenders / cities ────────────────────────────────────────────────────

#: The No-RS control. Always run, never a choice: it is the baseline every
#: paired comparison is defined against.
CONTROL = "No RS"

#: Welfare gates a built recommender may wear on top of its ranking.
WELFARE_GATES = ("pup", "rm", "pup_rm")

#: Ceiling on user-built recommenders in one study (the control is extra).
#: Cost and retained map geometry both scale linearly in arms.
MAX_RECOMMENDERS = 5

#: Ceiling on points in the spatial welfare map. Well past what a hexbin can
#: resolve, and it keeps the run payload bounded regardless of population.
WELFARE_MAP_MAX_POINTS = 20000

#: Deprecated fixed vocabulary, kept so saved API calls still resolve.
TREATMENTS = ["No RS", "Standard RS", "PUP", "RM", "PUP+RM"]

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


def precomputed_cities() -> dict:
    """``{metro key: routing matrices present}``.

    A warmed metro without matrices still runs — it just falls back to computing
    Dijkstra trees per run, which is what makes runs expensive and serialised.
    Read fresh each call so a metro precomputed while the service is up is
    reflected without a restart.
    """
    cache_dir = params.GEO_PARAMS["cache_dir"]
    out = {}
    for key, _ in CITIES:
        meta = routing_matrix.read_meta(cache_dir, key)
        out[key] = bool(meta and meta.get("version") == routing_matrix.MATRIX_VERSION)
    return out

MAX_SEEDS = 12

# Ceiling on simulations running at once. Each is a process holding its own copy
# of the metro graph (a few hundred MB resident); the multi-GB routing matrices
# are memmapped read-only, so those cost one shared page-cache copy no matter how
# many workers there are. Defaults to every core — set WELFARE_RS_MAX_WORKERS to
# leave the box headroom for other work.
MAX_WORKERS = int(os.environ.get("WELFARE_RS_MAX_WORKERS", "0") or 0) or (os.cpu_count() or 1)


def _worker_count(n_tasks: int) -> int:
    return max(1, min(n_tasks, os.cpu_count() or 1, MAX_WORKERS))


# ── Config / run records ─────────────────────────────────────────────────────

@dataclass(frozen=True)
class RecommenderSpec:
    """One study arm: a full configuration of the single unified scorer.

    Frozen and built only of scalars, so it is hashable (the run id folds it in)
    and picklable (it crosses into the pool workers as-is).
    """

    label: str = "Custom RS"
    w_rating: float = 0.20
    w_reviews: float = 0.15
    w_relevance: float = 0.10
    w_proximity: float = 0.25
    w_popularity: float = 0.20
    w_personalization: float = 0.10
    distance_scale_km: float = 5.0
    popularity_gamma: float = 1.0
    learning_rate: float = 0.12
    welfare_gate: str = ""          # "" | "pup" | "rm" | "pup_rm"
    pup_alpha: float = 0.6
    rm_epsilon: float = 0.3
    random_ranking: bool = False

    def to_engine_config(self) -> RecommenderConfig:
        return RecommenderConfig(
            label=self.label,
            w_rating=self.w_rating, w_reviews=self.w_reviews,
            w_relevance=self.w_relevance, w_proximity=self.w_proximity,
            w_popularity=self.w_popularity,
            w_personalization=self.w_personalization,
            distance_scale_km=self.distance_scale_km,
            popularity_gamma=self.popularity_gamma,
            learning_rate=self.learning_rate,
            random_ranking=self.random_ranking,
        )

    def identity(self) -> tuple:
        """Everything that changes behaviour, for the run-id hash.

        The label is included: two arms differing only by name are two rows in
        the results, so they must not collapse to one cached study.
        """
        return (
            self.label,
            round(self.w_rating, 4), round(self.w_reviews, 4),
            round(self.w_relevance, 4), round(self.w_proximity, 4),
            round(self.w_popularity, 4), round(self.w_personalization, 4),
            round(self.distance_scale_km, 4), round(self.popularity_gamma, 4),
            round(self.learning_rate, 4), self.welfare_gate,
            round(self.pup_alpha, 4), round(self.rm_epsilon, 4),
            bool(self.random_ranking),
        )


def spec_from_legacy_name(
    name: str, pup_alpha: float, rm_epsilon: float
) -> Optional[RecommenderSpec]:
    """Map a pre-builder treatment name onto the equivalent spec.

    "Standard RS" was the four-platform stack with default weights; the
    balanced preset is its closest single-scorer equivalent. The gated
    treatments are that same ranking with the welfare filter switched on, which
    is exactly what they always were.
    """
    if name == CONTROL:
        return None
    gate = {"PUP": "pup", "RM": "rm", "PUP+RM": "pup_rm"}.get(name, "")
    base = dict(RECOMMENDER_PRESETS["Balanced hybrid"])
    return RecommenderSpec(
        label=name, **base,
        welfare_gate=gate, pup_alpha=pup_alpha, rm_epsilon=rm_epsilon,
    )


@dataclass(frozen=True)
class RunConfig:
    city: str = DEFAULT_CITY
    num_agents: int = 80
    num_days: int = 30
    seed: int = 42          # base seed; the study sweeps seed .. seed+num_seeds-1
    num_seeds: int = 3
    # Recommenders to run *besides* the No-RS control, which is always included.
    # A tuple, not a list, so the frozen dataclass stays hashable.
    recommenders: Tuple[RecommenderSpec, ...] = ()
    multimodal: bool = False
    use_real_pois: bool = True

    def seeds(self) -> List[int]:
        n = max(1, min(int(self.num_seeds), MAX_SEEDS))
        return [int(self.seed) + i for i in range(n)]

    def all_conditions(self) -> List[str]:
        """``[CONTROL, *built recommenders]`` in the order the user built them.

        The control is always present and always first: every Table-2 row is
        defined relative to it. Order is the user's, not canonical — with
        free-form labels there is no canonical order to impose, and the run id
        folds in the sequence so a reordered study is a different (correctly
        re-run) study rather than a stale cache hit.
        """
        return [CONTROL] + [s.label for s in self.recommenders]

    def run_id(self) -> str:
        key = (
            self.city, int(self.num_agents), int(self.num_days), int(self.seed),
            int(self.num_seeds), tuple(s.identity() for s in self.recommenders),
            bool(self.multimodal), bool(self.use_real_pois),
        )
        return hashlib.sha1(repr(key).encode()).hexdigest()[:16]


@dataclass
class SeedViz:
    """One condition's geometry at one seed — what the map plays back.

    POIs are deliberately *not* held here. The catalog varies with the seed (in
    synthetic mode) but never with which recommender ran, so it lives once per
    seed on the Run rather than being duplicated across every condition.
    """
    seed: int
    merged: Dict[int, dict]
    trip_index: Dict[str, dict]
    time_span: int


@dataclass
class Run:
    run_id: str
    config: RunConfig
    seeds: List[int]
    conditions: List[str]                        # No RS first, then recommenders
    per_condition: Dict[str, Dict[int, SeedViz]]  # condition -> seed -> geometry
    pois_by_seed: Dict[int, List[dict]]          # condition-independent
    table1: List[dict]            # one row per condition (No RS first)
    table2: List[dict]            # one row per recommender, paired vs No RS
    headlines: Dict[str, dict]    # condition -> headline metrics
    aggregate: Dict[str, dict]    # condition -> acceptance / travel-time summary
    metrics: Dict[str, dict]      # condition -> dashboard panels
    welfare_map: List[dict]       # per-agent home + per-condition utility delta
    view: dict
    bounds: dict                  # full metro network extent
    core_bounds: dict             # principal-city extent (initial map frame)
    num_days: int
    num_intersections: int
    poi_count: int
    poi_source: str
    # Wall-clock cost of the study, filled in once it completes.
    created_at: Optional[float] = None
    duration_sec: Optional[float] = None

    def default_condition(self) -> str:
        """What the map opens on: the first recommender, else the No-RS control."""
        for c in self.conditions:
            if c != CONTROL:
                return c
        return CONTROL

    def meta(self) -> dict:
        """Aggregated metrics + per-condition/seed index (no geometry)."""
        return {
            "run_id": self.run_id,
            "config": asdict(self.config),
            "seeds": self.seeds,
            "default_seed": self.seeds[0] if self.seeds else self.config.seed,
            "conditions": self.conditions,
            "default_condition": self.default_condition(),
            # {condition: {seed: span}} — the scrubber's extent depends on both.
            "time_spans": {
                cond: {str(s): sv.time_span for s, sv in by_seed.items()}
                for cond, by_seed in self.per_condition.items()
            },
            "table1": self.table1,
            "table2": self.table2,
            "headlines": self.headlines,
            "aggregate": self.aggregate,
            "metrics": self.metrics,
            "welfare_map": self.welfare_map,
            "income_bands": sim_metrics.income_band_order(),
            "trust_groups": sim_metrics.trust_group_labels(),
            "view": self.view,
            "bounds": self.bounds,
            "core_bounds": self.core_bounds,
            "num_days": self.num_days,
            "num_intersections": self.num_intersections,
            "poi_count": self.poi_count,
            "poi_source": self.poi_source,
            "city_label": _CITY_LABELS.get(self.config.city, self.config.city),
            "created_at": self.created_at,
            "duration_sec": self.duration_sec,
        }

    def viz(self, condition: Optional[str], seed: Optional[int]) -> Optional[SeedViz]:
        """Geometry for (condition, seed); unknown values fall back to defaults."""
        by_seed = None
        if condition is not None:
            by_seed = self.per_condition.get(condition)
        if by_seed is None:
            by_seed = self.per_condition.get(self.default_condition())
        if not by_seed:
            return None
        if seed is not None and int(seed) in by_seed:
            return by_seed[int(seed)]
        return by_seed.get(self.seeds[0]) if self.seeds else None

    def pois_for(self, seed: Optional[int]) -> List[dict]:
        """The POI catalog for a seed (identical across conditions)."""
        if seed is not None and int(seed) in self.pois_by_seed:
            return self.pois_by_seed[int(seed)]
        if self.seeds and self.seeds[0] in self.pois_by_seed:
            return self.pois_by_seed[self.seeds[0]]
        return next(iter(self.pois_by_seed.values()), [])


# ── Job (background study execution + progress) ──────────────────────────────

@dataclass
class Job:
    run_id: str
    status: str = "running"  # running | done | error | cancelled
    # Progress is counted in simulated DAYS across the whole (seed x condition)
    # grid — the finest unit the parent can observe now that a task is a whole
    # simulation. `per_condition` breaks that down per arm (days done, out of
    # `cond_total` = seeds x days each), because every condition is running at
    # the same time: a single "most recent tick" label would show one arm at
    # random and read as if the study were sequential.
    current: int = 0
    total: int = 0
    per_condition: Dict[str, int] = field(default_factory=dict)
    cond_total: int = 0
    error: Optional[str] = None
    # Retained so /api/runs can describe a study that is still computing (and
    # therefore has no Run record yet), and so a browser that never saw the
    # submission can still show what it is.
    config: Optional["RunConfig"] = None
    created_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
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

    def snapshot(self) -> dict:
        """Status + progress only (no geometry), for the run list."""
        with self._lock:
            return {
                "status": self.status,
                "created_at": self.created_at,
                "finished_at": self.finished_at,
                "error": self.error,
                "progress": {
                    "current": self.current,
                    "total": self.total,
                    "cond_total": self.cond_total,
                    "per_condition": dict(self.per_condition),
                },
            }

    def finish(self, status: str) -> None:
        self.status = status
        self.finished_at = time.time()

    def subscribe(self) -> "queue.Queue":
        q: "queue.Queue" = queue.Queue()
        with self._lock:
            self._subs.append(q)
            if self.status == "running":
                q.put({"type": "progress", "current": self.current,
                       "total": self.total, "cond_total": self.cond_total,
                       "per_condition": dict(self.per_condition),
                       "elapsed_sec": time.time() - self.created_at})
            elif self.status == "done":
                q.put({"type": "done", "run_id": self.run_id})
            elif self.status == "error":
                q.put({"type": "error", "message": self.error or "run failed"})
            elif self.status == "cancelled":
                q.put({"type": "cancelled", "run_id": self.run_id})
        return q


# ── Recommender wiring (welfare gate) ────────────────────────────────────────

def _make_recommender_factory(spec: Optional[RecommenderSpec]):
    """Wrap the built recommender in its welfare gate, if it has one.

    Returns None when the spec is ungated (or absent), which leaves the
    simulation's own configurable recommender in place unwrapped.
    """
    if spec is None or spec.welfare_gate not in WELFARE_GATES:
        return None
    from welfare_rs import TravelCostEstimator, WelfareAwareOrchestrator

    mode = spec.welfare_gate
    pup_alpha = spec.pup_alpha
    rm_epsilon = spec.rm_epsilon

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


# ── Worker: run ONE simulation (one seed, one condition) ─────────────────────
#
# Module-level so it is picklable by ProcessPoolExecutor. Each worker keeps the
# (POI-augmented) network for a (city, use_real_pois) in a process-global cache
# and reuses it across every task it handles; the geometry is independent of both
# seed and condition.

_WORKER_NET_CACHE: Dict[tuple, object] = {}


def _spec_for(cfg: dict, condition: str) -> Optional[RecommenderSpec]:
    """The spec driving ``condition``, or None for the control.

    Specs travel into the worker as plain dicts (``asdict`` on the config), so
    they are rebuilt here rather than unpickled as dataclasses — keeping the
    task payload to primitives.
    """
    if condition == CONTROL:
        return None
    for raw in cfg.get("recommenders", ()):
        if raw.get("label") == condition:
            return RecommenderSpec(**raw)
    return None


def _worker_network(city: str, use_real_pois: bool):
    key = (city, bool(use_real_pois))
    net = _WORKER_NET_CACHE.get(key)
    if net is None:
        net = build_network(city)  # metro two-layer or legacy preset
        _WORKER_NET_CACHE[key] = net
    return net


def _build_sim(cfg: dict, seed: int, condition: str, network, poi_rows,
               spec: Optional[RecommenderSpec]):
    ds = get_datasource()
    return Simulation(
        num_agents=int(cfg["num_agents"]),
        seed=int(seed),
        use_recommenders=(condition != CONTROL),
        # Population from the datasource: under WELFARE_RS_DATASOURCE=postgres
        # this is the survey.personas view, otherwise the persona CSV. The path
        # is still passed as the fallback for a worker whose datasource has
        # neither (personas() returns None).
        persona_rows=ds.personas(),
        persona_csv_path=ds.persona_csv_path() or params.SURVEY_PERSONA_CSV_PATH,
        road_network=network,
        poi_rows=poi_rows,
        disabled_modes=("transit",),
        # The control still builds a recommender it never consults, so that the
        # catalog, tastes and RNG streams are identical to the treated arms;
        # `use_recommenders=False` is what actually withholds it from agents.
        recommender_config=(spec.to_engine_config() if spec is not None else None),
        recommender_factory=_make_recommender_factory(spec),
    )


def _run_one(cfg: dict, seed: int, condition: str, cancel_event=None,
             progress_q=None) -> dict:
    """Run ONE simulation: one condition at one seed. The unit of parallelism.

    Returns a picklable dict of that run's metrics, its per-agent utilities and
    its viz payload. ``utils`` is the piece that makes the fan-out possible:
    Table 2 pairs a recommender against No RS agent-by-agent, and shipping the
    ``{agent_id: eval-day leisure utility}`` map back lets the parent do that
    pairing across process boundaries (one float per agent).

    Raises SimulationCancelled if ``cancel_event`` is set between days.
    """
    # Mode model: paper default is car-only; the toggle re-enables multimodal.
    # Safe to set on the module global — this is the worker's own process.
    params.SIMPLIFICATION_TOGGLES["CAR_ONLY_MODE"] = not bool(cfg["multimodal"])

    poi_rows = get_datasource().poi_rows(cfg["city"]) if bool(cfg["use_real_pois"]) else None
    use_real_pois = bool(poi_rows)
    network = _worker_network(cfg["city"], use_real_pois)

    # Cooperative cancellation: checked between days by the engine. `cancel_event`
    # is the Job's threading.Event (inline) or a shared mp Event (pool workers).
    should_stop = (lambda: cancel_event.is_set()) if cancel_event is not None else None

    per_day: List[Dict[int, dict]] = []
    spec = _spec_for(cfg, condition)
    sim = _build_sim(cfg, seed, condition, network, poi_rows, spec)
    # Dashboard panels accumulate across every day (see metrics.RunMetrics);
    # only Table 1 / Table 2 stay eval-day quantities.
    run_metrics = sim_metrics.RunMetrics(sim)

    def _on_day(day_index, sim_) -> None:
        per_day.append(viz.build_timelines(sim_))
        run_metrics.observe_day(sim_)
        # Day-granular progress. With one simulation per task there is no other
        # way for the parent to see inside a task, and tasks are long.
        if progress_q is not None:
            try:
                progress_q.put({"seed": int(seed), "condition": condition,
                                "day": int(day_index) + 1})
            except Exception:
                pass  # progress is advisory — never fail a run over it

    sim.run_days(int(cfg["num_days"]), on_day_complete=_on_day, should_stop=should_stop)

    merged = viz.merge_day_timelines(per_day)
    summ = sim.summarize()
    out = {
        "seed": int(seed),
        "condition": condition,
        "t1": table1_metrics(sim),
        "utils": eval_day_leisure_utilities(sim.agents),
        "metrics": run_metrics.finalize(sim),
        "summary": {
            "recommendation_acceptance_rate": summ.get("recommendation_acceptance_rate", 0.0),
            "avg_travel_time_min": summ.get("avg_travel_time_min", 0.0),
        },
        "viz": {
            "merged": merged,
            "trip_index": viz.build_trip_index(merged),
            "time_span": viz.time_span(merged),
        },
    }

    # The POI catalog, the agent population and the network extent are identical
    # across conditions at a given seed, and the control always runs — so ship
    # them back from that task alone instead of pickling them once per arm.
    if condition == CONTROL:
        out["agents"] = sim_metrics.agent_frame(sim)
        net = sim.road_network
        south, west, north, east = net.bounds
        csouth, cwest, cnorth, ceast = net.core_bounds
        out["pois"] = [
            {"lon": round(p.location[1], 6), "lat": round(p.location[0], 6),
             "category": p.category, "label": f"{p.name} · {p.category}"}
            for p in sim.place_catalog
        ]
        out["net"] = {
            # Initial map frame: the principal city; agents commute in from
            # the wider metro bounds visible on zoom-out.
            "view": {"latitude": net.core_center[0], "longitude": net.core_center[1]},
            "bounds": {"south": south, "west": west, "north": north, "east": east},
            "core_bounds": {"south": csouth, "west": cwest, "north": cnorth, "east": ceast},
            "num_intersections": net.num_base_nodes,
            "poi_count": len(sim.place_catalog),
            "poi_source": "real dataset" if use_real_pois else "synthetic",
        }
    return out


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


def _agg_table2(condition: str, per_seed_t2: List[dict]) -> dict:
    return {
        "condition": condition,
        "harmed_pct": float(np.mean([t["harmed_pct"] for t in per_seed_t2])),
        "improved_pct": float(np.mean([t["improved_pct"] for t in per_seed_t2])),
        "mean_orc": float(np.mean([t["mean_orc"] for t in per_seed_t2])),
        "n_matched": int(np.sum([t["n_matched"] for t in per_seed_t2])),
    }


def _pooled_mean(rows: List[dict], value_key: str, weight_key: str = "n") -> float:
    """Sample-size-weighted mean across seeds.

    Weighted, not a mean of means: a seed where only nine agents in a band took
    a leisure trip would otherwise count as much as one where sixty did, which
    for the thinner income bands is most of the noise in the panel.
    """
    total = sum(float(r.get(weight_key, 0) or 0) for r in rows)
    if total <= 0:
        return 0.0
    return sum(float(r.get(value_key, 0.0)) * float(r.get(weight_key, 0) or 0)
               for r in rows) / total


def _agg_segment_panel(per_seed: List[dict]) -> Dict[str, dict]:
    """Merge one segmentation's ``{group: {n, mean_utility, neg_rate}}`` maps."""
    groups = {g for seed_rows in per_seed for g in seed_rows}
    out = {}
    for group in sorted(groups):
        rows = [s[group] for s in per_seed if group in s]
        out[group] = {
            "n": int(sum(r["n"] for r in rows)),
            "mean_utility": _pooled_mean(rows, "mean_utility"),
            "neg_rate": _pooled_mean(rows, "neg_rate"),
        }
    return out


def _agg_spatial(per_seed: List[dict]) -> Dict[str, dict]:
    """Merge the spatial panel across seeds."""
    def merge(rows: List[dict]) -> dict:
        return {
            "n": int(sum(r["n"] for r in rows)),
            "mean_distance_km": _pooled_mean(rows, "mean_distance_km"),
            "mean_emissions_g": _pooled_mean(rows, "mean_emissions_g"),
            "mean_travel_min": _pooled_mean(rows, "mean_travel_min"),
            # A median cannot be pooled from per-seed medians; the weighted mean
            # of them is the honest approximation and is labelled as such.
            "median_detour": _pooled_mean(rows, "median_detour"),
            "median_excess_km": _pooled_mean(rows, "median_excess_km"),
        }

    bands = {b for s in per_seed for b in s["by_income"]}
    return {
        "overall": merge([s["overall"] for s in per_seed]),
        "by_income": {
            band: merge([s["by_income"][band] for s in per_seed if band in s["by_income"]])
            for band in sorted(bands)
        },
    }


def _agg_per_day(per_seed: List[List[dict]]) -> List[dict]:
    """Average each day's record across seeds, truncated to the shortest run."""
    if not per_seed:
        return []
    length = min(len(s) for s in per_seed)
    keys = sorted({k for s in per_seed for row in s for k in row})
    out = []
    for day in range(length):
        row = {"day": day + 1}
        for key in keys:
            values = [s[day][key] for s in per_seed if key in s[day]]
            row[key] = float(np.mean(values)) if values else 0.0
        out.append(row)
    return out


def _agg_lorenz(curves: List[List[list]]) -> List[list]:
    """Average Lorenz curves pointwise.

    ``lorenz_points`` normally samples a fixed 41-point x-grid, which makes a
    pointwise mean well defined — but its no-visits early return emits just two
    points. A seed in which nobody took a leisure trip would therefore be a
    short curve, and zipping it against the others would run off its end. Only
    curves on the majority grid are averaged; the rest are dropped rather than
    interpolated, since a two-point curve carries no shape to preserve.
    """
    if not curves:
        return []
    widths = [len(c) for c in curves]
    grid = max(set(widths), key=widths.count)
    usable = [c for c in curves if len(c) == grid]
    if not usable:
        return []
    return [
        [usable[0][i][0], float(np.mean([c[i][1] for c in usable]))]
        for i in range(grid)
    ]


def _agg_metrics(per_seed: List[dict]) -> Dict[str, object]:
    """Everything the dashboard needs for one condition, merged over seeds."""
    category: Dict[str, Dict[str, int]] = {}
    for seed_rows in per_seed:
        for band, by_category in seed_rows["category_by_income"].items():
            row = category.setdefault(band, {})
            for name, count in by_category.items():
                row[name] = row.get(name, 0) + count

    segmentations = {k for s in per_seed for k in s["segments"]}
    lorenz = [s["lorenz"] for s in per_seed]
    return {
        "per_day": _agg_per_day([s["per_day"] for s in per_seed]),
        "segments": {
            key: _agg_segment_panel([s["segments"][key] for s in per_seed if key in s["segments"]])
            for key in sorted(segmentations)
        },
        "category_by_income": category,
        "spatial": _agg_spatial([s["spatial"] for s in per_seed]),
        "lorenz": _agg_lorenz(lorenz),
        "footfall": {
            key: float(np.mean([s["footfall"][key] for s in per_seed]))
            for key in per_seed[0]["footfall"]
        } if per_seed else {},
        "taste": {
            key: float(np.mean([s["taste"][key] for s in per_seed]))
            for key in per_seed[0]["taste"]
        } if per_seed else {},
    }


# ── Run manager ──────────────────────────────────────────────────────────────

class _InlineProgress:
    """A ``.put()``-compatible progress sink for the single-task inline path,
    where there is no process boundary and no Manager queue to route through."""

    def __init__(self, tick):
        self._tick = tick

    def put(self, event) -> None:
        self._tick(event)


class RunManager:
    """Builds and caches studies; serialises study execution (one study at a
    time), parallelising the whole (seed x condition) grid across processes."""

    # Studies now retain map geometry for EVERY condition, not just one, so a
    # cached study is several times larger than it used to be. Keep fewer.
    def __init__(self, max_runs: int = 4):
        # One study at a time keeps memory bounded and avoids oversubscribing the
        # CPU (seeds already fan out across all cores inside a study).
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="study")
        self._runs: Dict[str, Run] = {}
        self._jobs: Dict[str, Job] = {}
        self._access: Dict[str, float] = {}   # run_id -> last read, for LRU eviction
        self._lock = threading.Lock()
        self._max_runs = max_runs

    def pois_available(self) -> bool:
        return pois_available()

    def get_run(self, run_id: str) -> Optional[Run]:
        run = self._runs.get(run_id)
        if run is not None:
            # Touch on every read so an open tab keeps its study alive.
            self._access[run_id] = time.time()
        return run

    def list_runs(self) -> List[dict]:
        """Every run this process knows about, newest first — no geometry.

        Union of jobs (which exist from submission, so a study still computing
        appears immediately) and completed runs. ``available`` says whether the
        result is still resident: a study can be 'done' yet evicted, in which
        case it is listed but cannot be opened.
        """
        with self._lock:
            jobs = dict(self._jobs)
            resident = set(self._runs)
            runs = dict(self._runs)

        out: List[dict] = []
        for run_id, job in jobs.items():
            snap = job.snapshot()
            cfg = job.config or (runs[run_id].config if run_id in runs else None)
            out.append({
                "run_id": run_id,
                "available": run_id in resident,
                "city": cfg.city if cfg else None,
                "city_label": _CITY_LABELS.get(cfg.city, cfg.city) if cfg else None,
                "conditions": list(cfg.all_conditions()) if cfg else [],
                "num_agents": cfg.num_agents if cfg else None,
                "num_days": cfg.num_days if cfg else None,
                "num_seeds": cfg.num_seeds if cfg else None,
                **snap,
            })
        # A run whose job was pruned would otherwise vanish from the list.
        for run_id, run in runs.items():
            if run_id in jobs:
                continue
            out.append({
                "run_id": run_id, "available": True, "status": "done",
                "city": run.config.city,
                "city_label": _CITY_LABELS.get(run.config.city, run.config.city),
                "conditions": list(run.conditions),
                "num_agents": run.config.num_agents, "num_days": run.config.num_days,
                "num_seeds": run.config.num_seeds,
                "created_at": None, "finished_at": None, "error": None,
                "progress": {"current": 0, "total": 0, "cond_total": 0, "per_condition": {}},
            })
        out.sort(key=lambda r: (r.get("created_at") or 0.0), reverse=True)
        return out

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
            job = Job(run_id=run_id, total=len(config.seeds()), config=config)
            self._jobs[run_id] = job
        self._executor.submit(self._execute, run_id, config)
        return run_id, "running"

    def _execute(self, run_id: str, config: RunConfig) -> None:
        job = self._jobs[run_id]
        try:
            run = self._build_study(run_id, config, job)
            run.created_at = job.created_at
            run.duration_sec = time.time() - job.created_at
            with self._lock:
                self._runs[run_id] = run
                self._access[run_id] = time.time()
                while len(self._runs) > self._max_runs:
                    # Least-recently-ACCESSED, not oldest-inserted. Now that runs
                    # are listable and linkable (?run=<id>), the oldest study is
                    # quite likely the one someone left open in a tab; dropping it
                    # from under them would 410 a view they are actively using.
                    victim = min(self._runs, key=lambda r: self._access.get(r, 0.0))
                    self._runs.pop(victim, None)
                    self._access.pop(victim, None)
            job.finish("done")
            job.publish({"type": "done", "run_id": run_id})
        except SimulationCancelled:
            job.finish("cancelled")
            job.publish({"type": "cancelled", "run_id": run_id})
        except Exception as exc:
            job.error = f"{type(exc).__name__}: {exc}"
            job.finish("error")
            job.publish({"type": "error", "message": job.error})

    def _build_study(self, run_id: str, cfg: RunConfig, job: Job) -> Run:
        seeds = cfg.seeds()
        conditions = cfg.all_conditions()
        # One task = one simulation = one (seed, condition) pair, No RS included.
        tasks = [(s, c) for s in seeds for c in conditions]
        num_days = max(1, int(cfg.num_days))
        job.total = len(tasks) * num_days   # progress counted in simulated days
        # Each condition runs once per seed, so every arm has the same
        # denominator. Seeded at zero up front so the UI lists all the arms the
        # moment the study starts, rather than having them pop in one by one.
        cond_total = len(seeds) * num_days
        job.cond_total = cond_total
        job.per_condition = {c: 0 for c in conditions}

        # Pre-build the network once so worker processes only ever *load* the
        # disk caches (no concurrent Overpass download / warm-pickle race).
        build_network(cfg.city)

        cfg_dict = asdict(cfg)
        results: Dict[Tuple[int, str], dict] = {}

        # Ticks arrive from every worker at once, so the counter needs a lock.
        # Publishing happens outside it to keep the SSE fan-out off the critical
        # section.
        tick_lock = threading.Lock()

        def _tick(event: dict) -> None:
            with tick_lock:
                job.current = min(job.current + 1, job.total)
                cond = event.get("condition")
                if cond in job.per_condition:
                    job.per_condition[cond] = min(job.per_condition[cond] + 1, cond_total)
                # Snapshot under the lock; the publish below must not race a
                # concurrent worker mutating the dict mid-serialisation.
                cur, by_cond = job.current, dict(job.per_condition)
            # Elapsed is computed server-side and advanced locally by the client,
            # so the timer is immune to clock skew between browser and box.
            job.publish({"type": "progress", "current": cur, "total": job.total,
                         "cond_total": cond_total, "per_condition": by_cond,
                         "elapsed_sec": time.time() - job.created_at})

        if len(tasks) == 1:
            # Inline: the Job's threading.Event is visible to should_stop directly.
            s, c = tasks[0]
            res = _run_one(cfg_dict, s, c, cancel_event=job.cancel,
                           progress_q=_InlineProgress(_tick))
            results[(res["seed"], res["condition"])] = res
        else:
            workers = _worker_count(len(tasks))
            # 'spawn': fork-after-threads under uvicorn is unsafe.
            ctx = mp.get_context("spawn")
            # Manager-backed Event and Queue are shareable with spawned workers.
            # Setting the Event makes each worker's should_stop fire at its next
            # day boundary; the Queue carries day-level progress back.
            manager_proc = ctx.Manager()
            job.mp_cancel = manager_proc.Event()
            progress_q = manager_proc.Queue()
            if job.cancel.is_set():  # cancel arrived during setup
                job.mp_cancel.set()

            stop_drain = threading.Event()

            def _drain() -> None:
                while not stop_drain.is_set():
                    try:
                        _tick(progress_q.get(timeout=0.2))
                    except queue.Empty:
                        continue
                    except Exception:
                        break  # manager torn down — expected at end of study

            drain = threading.Thread(target=_drain, name="progress", daemon=True)
            drain.start()
            try:
                with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
                    futs = {
                        pool.submit(_run_one, cfg_dict, s, c, job.mp_cancel, progress_q): (s, c)
                        for s, c in tasks
                    }
                    try:
                        for fut in as_completed(futs):
                            res = fut.result()
                            results[(res["seed"], res["condition"])] = res
                    except SimulationCancelled:
                        for f in futs:
                            f.cancel()  # drop tasks not yet started
                        raise
            finally:
                stop_drain.set()
                drain.join(timeout=1.0)
                job.mp_cancel = None
                manager_proc.shutdown()

        return self._assemble(run_id, cfg, seeds, results)

    @staticmethod
    def _welfare_map(conditions: List[str], ok_seeds: List[int],
                     results: Dict[Tuple[int, str], dict]) -> List[dict]:
        """Per-agent home point carrying each arm's utility delta vs the control.

        One row per (seed, agent) that took a leisure trip under the control and
        under at least one recommender; ``delta`` holds ``condition -> ΔU``. The
        map bins these into hexagons, so the payload is deliberately flat and
        rounded rather than pre-aggregated — the frontend chooses the bin size.

        Built from each arm's ``run_utility`` (mean leisure net utility over the
        WHOLE run), not from the eval-day ``utils`` that Table 2 uses. Table 2 is
        a paper-defined eval-day quantity and stays that way; a map is not, and
        on a single day only a minority of agents go out, so intersecting "went
        out under the control" with "went out under this arm" left barely a
        couple of dozen points to bin.

        Agents with no leisure trip in an arm are simply absent from that arm's
        delta rather than recorded as zero: a zero would read as "unaffected"
        when what actually happened is "did not go out", and averaged into a
        hexagon it would pull the cell toward neutral.

        Capped at ``WELFARE_MAP_MAX_POINTS``: this is the one part of the run
        payload that scales with agents x seeds, and at the top of the allowed
        range (10,000 agents, 12 seeds) it would be a 120,000-row, ~12 MB JSON
        body on every /api/runs/{id}. Over the cap it is thinned by an even
        stride rather than by truncation, which would otherwise drop whole seeds
        and, with them, whole parts of the map.
        """
        rows: List[dict] = []
        for seed in ok_seeds:
            frame = results[(seed, CONTROL)].get("agents")
            if not frame:
                continue
            control = results[(seed, CONTROL)]["metrics"]["run_utility"]
            for i, agent_id in enumerate(frame["agent_id"]):
                base = control.get(agent_id)
                if base is None:
                    continue
                deltas = {}
                for condition in conditions:
                    if condition == CONTROL:
                        continue
                    value = results[(seed, condition)]["metrics"]["run_utility"].get(agent_id)
                    if value is not None:
                        deltas[condition] = round(float(value) - float(base), 5)
                if not deltas:
                    continue
                rows.append({
                    "lat": frame["home_lat"][i],
                    "lon": frame["home_lon"][i],
                    "income": frame["income"][i],
                    "delta": deltas,
                })
        if len(rows) > WELFARE_MAP_MAX_POINTS:
            stride = math.ceil(len(rows) / WELFARE_MAP_MAX_POINTS)
            rows = rows[::stride]
        return rows

    def _assemble(self, run_id: str, cfg: RunConfig, seeds: List[int],
                  results: Dict[Tuple[int, str], dict]) -> Run:
        conditions = cfg.all_conditions()
        # Only seeds that produced *every* condition are usable: a seed missing
        # its No-RS run has nothing to pair against, and one missing a
        # recommender would silently drop that arm from a single row of the
        # cross-seed average.
        ok_seeds = [s for s in seeds if all((s, c) in results for c in conditions)]
        if not ok_seeds:
            raise RuntimeError("study produced no complete seed")

        # Table 1: one row per condition, No RS first (all_conditions order).
        table1 = [
            _agg_table1_row(c, [results[(s, c)]["t1"] for s in ok_seeds])
            for c in conditions
        ]

        # Table 2: each recommender paired against the SAME No-RS run, seed by
        # seed, then averaged. The pairing lives here rather than in the worker
        # because the two runs are now in different processes.
        table2 = [
            _agg_table2(c, [
                table2_from_utilities(results[(s, c)]["utils"],
                                      results[(s, CONTROL)]["utils"])
                for s in ok_seeds
            ])
            for c in conditions if c != CONTROL
        ]

        t1_by_cond = {r["condition"]: r for r in table1}
        t2_by_cond = {r["condition"]: r for r in table2}
        # One headline per condition; the frontend shows whichever the map is on.
        # The No-RS control has no Table-2 row (it is the baseline), so its
        # over-recommendation fields are null rather than zero.
        headlines = {}
        for c in conditions:
            t1r, t2r = t1_by_cond[c], t2_by_cond.get(c)
            headlines[c] = {
                "mean_utility": t1r["mean_utility"],
                "sigma_u": t1r["sigma_u"],
                "neg_rate": t1r["neg_rate"],
                "harmed_pct": t2r["harmed_pct"] if t2r else None,
                "improved_pct": t2r["improved_pct"] if t2r else None,
                "mean_orc": t2r["mean_orc"] if t2r else None,
            }

        aggregate = {
            c: {
                "acceptance_rate": float(np.mean([
                    results[(s, c)]["summary"]["recommendation_acceptance_rate"]
                    for s in ok_seeds])),
                "avg_travel_time_min": float(np.mean([
                    results[(s, c)]["summary"]["avg_travel_time_min"]
                    for s in ok_seeds])),
            }
            for c in conditions
        }

        metrics = {
            c: _agg_metrics([results[(s, c)]["metrics"] for s in ok_seeds])
            for c in conditions
        }
        welfare_map = self._welfare_map(conditions, ok_seeds, results)

        per_condition = {
            c: {
                s: SeedViz(
                    seed=s,
                    merged={int(k): v for k, v in results[(s, c)]["viz"]["merged"].items()},
                    trip_index=results[(s, c)]["viz"]["trip_index"],
                    time_span=results[(s, c)]["viz"]["time_span"],
                )
                for s in ok_seeds
            }
            for c in conditions
        }
        # POIs and the network extent came back on the No-RS task only.
        pois_by_seed = {s: results[(s, CONTROL)].get("pois", []) for s in ok_seeds}
        net0 = results[(ok_seeds[0], CONTROL)]["net"]

        return Run(
            run_id=run_id,
            config=cfg,
            seeds=ok_seeds,
            conditions=conditions,
            per_condition=per_condition,
            pois_by_seed=pois_by_seed,
            table1=table1,
            table2=table2,
            headlines=headlines,
            aggregate=aggregate,
            metrics=metrics,
            welfare_map=welfare_map,
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

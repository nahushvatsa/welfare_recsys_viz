"""Dashboard metrics, computed inside the simulation worker.

Every study arm runs in its own process, and results cross that boundary by
pickle. Shipping per-trip records back would put the whole run on the wire; these
functions therefore reduce a finished :class:`~welfare_rs.simulation.Simulation`
to a few hundred numbers per arm, and the parent only ever assembles.

Two timescales, matching how the paper's tables already work:

* :func:`day_metrics` runs at the end of every simulated day (via the engine's
  ``on_day_complete`` hook), while ``agent.trips`` still holds that day's trips —
  the next day's planning clears them. This is what the by-day curves are made of.
* :func:`eval_day_metrics` runs once at the end, over the final (evaluation)
  day's trips, matching ``experiment_harness.table1_metrics``.

Nothing here mutates the simulation.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import params, tastes
from .experiment_harness import gini_coefficient, leisure_net_utility

_EARTH_RADIUS_KM = 6371.0088

#: Trust in platforms is cut into this many equal-count groups. Quartiles are
#: enough to show a monotone gradient without splintering the per-group counts.
TRUST_GROUPS = 4


# ── Segmentation ─────────────────────────────────────────────────────────────

def income_band(income: float) -> str:
    """Persona income band containing ``income``.

    Matched on lower bound rather than by testing ``lo <= x <= hi``: the bands in
    ``PERSONA_MAPPING`` have gaps between them (34,000 to 35,000), and randomly
    generated agents draw from a continuous range that lands in those gaps.
    """
    bands = params.PERSONA_MAPPING["income_bands"]
    ordered = sorted(bands.items(), key=lambda kv: kv[1][0])
    label = ordered[0][0]
    for name, (low, _high) in ordered:
        if income >= low:
            label = name
    return label


def income_band_order() -> List[str]:
    """Bands poorest-first, for stable chart axes."""
    bands = params.PERSONA_MAPPING["income_bands"]
    return [k for k, _ in sorted(bands.items(), key=lambda kv: kv[1][0])]


def trust_group_labels() -> List[str]:
    return [f"Q{i + 1}" for i in range(TRUST_GROUPS)]


def _trust_groups(agents) -> Dict[int, str]:
    """``agent_id -> quartile label`` by trust in platforms.

    Quartiles are cut within the run's own population rather than on absolute
    thresholds, so the groups stay balanced whatever the persona mix. Ties are
    broken by agent id so the assignment is deterministic and identical across
    conditions — trust is an input, so agent i is in the same group in every arm.
    """
    scored = sorted(
        ((float(a.latent_variables.get("trust_platforms", 0.5)), a.id) for a in agents),
    )
    n = len(scored)
    out: Dict[int, str] = {}
    if n == 0:
        return out
    for rank, (_value, agent_id) in enumerate(scored):
        group = min(TRUST_GROUPS - 1, rank * TRUST_GROUPS // n)
        out[agent_id] = f"Q{group + 1}"
    return out


def agent_segments(sim) -> Dict[int, Dict[str, str]]:
    """``agent_id -> {segmentation: group}`` for every agent.

    ``paradigm`` is the agent's mode-choice decision rule. It acts only when
    the agent has more than one mode to choose between, so for agents who live
    outside the core (car only) it never fires.
    """
    trust = _trust_groups(sim.agents)
    return {
        a.id: {
            "income": income_band(a.characteristics["income"]),
            "trust": trust.get(a.id, "Q1"),
            "paradigm": a.decision_paradigms.get("mode_choice", "utility"),
        }
        for a in sim.agents
    }


# ── Footfall concentration ───────────────────────────────────────────────────

def footfall_metrics(sim) -> Dict[str, float]:
    """Concentration of leisure visits across the POI catalog.

    The headline is ``effective_places``, the inverse-Simpson number
    ``1 / Σ pᵢ²``: the count of equally-busy venues that would produce the
    observed spread. Ten thousand visits split evenly over 40 places gives 40;
    the same visits with half at one venue gives about 4.

    That statistic is here because the obvious ones do not survive this
    geometry. A metro catalog holds thousands of POIs while a study produces
    hundreds of visits, so almost every POI is a structural zero: the Gini over
    the full catalog pins near 0.99 for *every* condition and percentile shares
    like "top 5%" cover more places than were ever visited, reading 1.000
    everywhere. Both would be measuring catalog size, not recommender behaviour.
    ``effective_places`` and the absolute ``top10_share`` are unaffected by how
    many places nobody went to, and ``gini`` below is therefore taken over the
    *visited* places only — concentration among the venues actually in play.
    """
    visits = sim.place_dynamics.visits
    counts = np.array(
        [float(visits.get(p.place_id, 0)) for p in sim.place_catalog], dtype=float
    )
    n_places = len(counts)
    total = float(counts.sum())
    if n_places == 0 or total <= 0:
        return {
            "gini": 0.0, "effective_places": 0.0, "top10_share": 0.0,
            "distinct_visited": 0.0, "coverage": 0.0, "repeat_rate": 0.0,
            "total_visits": 0.0, "n_places": float(n_places),
        }

    nonzero = counts[counts > 0]
    shares = nonzero / total
    ordered = np.sort(nonzero)[::-1]
    return {
        "gini": gini_coefficient(nonzero) if len(nonzero) >= 2 else 0.0,
        "effective_places": float(1.0 / np.sum(shares ** 2)),
        "top10_share": float(ordered[:10].sum() / total),
        "distinct_visited": float(len(nonzero)),
        "coverage": float(len(nonzero) / n_places),
        # Share of visits that were not the first to their venue: how much of
        # the traffic is agents going back somewhere rather than exploring.
        "repeat_rate": float(1.0 - len(nonzero) / total),
        "total_visits": total,
        "n_places": float(n_places),
        # Visits per catalog POI. Reported so the dashboard can say when the
        # concentration panel is too thinly sampled to mean much: a metro
        # catalog holds thousands of POIs, and until a run produces visits on
        # that order most venues are structural zeros and no feedback loop can
        # ignite. Raising it means more agents or more days — the catalog cannot
        # simply be shrunk, because the POI selection determines the graph node
        # ids that index the precomputed routing matrices.
        "visits_per_place": float(total / n_places),
    }


def lorenz_points(sim, n_points: int = 41) -> List[List[float]]:
    """Lorenz curve of footfall as ``[[population share, visit share], ...]``.

    Downsampled to ``n_points`` so the payload is a fixed small size regardless
    of catalog size; the endpoints (0,0) and (1,1) are always included.
    """
    visits = sim.place_dynamics.visits
    counts = np.sort(
        np.array([float(visits.get(p.place_id, 0)) for p in sim.place_catalog])
    )
    total = float(counts.sum())
    if len(counts) == 0 or total <= 0:
        return [[0.0, 0.0], [1.0, 1.0]]
    cumulative = np.cumsum(counts) / total
    xs = np.linspace(0.0, 1.0, n_points)
    idx = np.clip((xs * len(counts)).astype(int) - 1, 0, len(counts) - 1)
    ys = np.where(xs <= 0.0, 0.0, cumulative[idx])
    return [[round(float(x), 4), round(float(y), 4)] for x, y in zip(xs, ys)]


# ── Spatial ──────────────────────────────────────────────────────────────────

def _haversine_vec(origin: Tuple[float, float], lats: np.ndarray, lons: np.ndarray):
    """Great-circle km from one point to arrays of points."""
    lat1 = math.radians(origin[0])
    lon1 = math.radians(origin[1])
    lat2 = np.radians(lats)
    lon2 = np.radians(lons)
    h = (
        np.sin((lat2 - lat1) / 2.0) ** 2
        + math.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2.0) ** 2
    )
    return 2.0 * _EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(h, 0.0, 1.0)))


class _NearestPOI:
    """Nearest catalog POI of a subtype, by straight-line distance."""

    def __init__(self, sim):
        self._by_subtype = {}
        for subtype, places in sim._pois_by_subtype.items():
            if not places:
                continue
            self._by_subtype[subtype] = (
                np.array([p.location[0] for p in places], dtype=float),
                np.array([p.location[1] for p in places], dtype=float),
            )

    def distance_km(self, subtype: str, origin: Tuple[float, float]) -> Optional[float]:
        arrays = self._by_subtype.get(subtype)
        if arrays is None:
            return None
        lats, lons = arrays
        return float(_haversine_vec(origin, lats, lons).min())


# ── Taste discovery ──────────────────────────────────────────────────────────

def taste_metrics(sim) -> Dict[str, float]:
    """Share of this day's leisure visits that matched the agent's latent taste.

    Split by how the destination was reached. ``recommended`` is the number that
    matters: the platform never observes taste, so any rise across days is the
    personalization channel inferring it from feedback. ``organic`` is the
    agent's own proximity-weighted pick and is the natural control — it should
    stay flat, because nothing about it learns.
    """
    counters = {"all": [0, 0], "recommended": [0, 0], "organic": [0, 0]}
    for agent in sim.agents:
        for trip in agent.trips:
            if trip.purpose != "leisure" or not trip.place_id:
                continue
            matched = 1 if sim.taste_bonus(agent, trip.place_id) > 0.0 else 0
            lane = "recommended" if trip.accepted_recommendation else "organic"
            for key in ("all", lane):
                counters[key][0] += matched
                counters[key][1] += 1
    return {
        f"{key}_rate": (hits / total if total else 0.0)
        for key, (hits, total) in counters.items()
    } | {f"{key}_n": total for key, (_hits, total) in counters.items()}


# ── Per-day and end-of-run bundles ───────────────────────────────────────────

def day_metrics(sim) -> Dict[str, float]:
    """Compact per-day record. Called while the day's trips are still live."""
    leisure = sum(
        1 for a in sim.agents for t in a.trips if t.purpose == "leisure"
    )
    accepted = sum(
        1
        for a in sim.agents
        for t in a.trips
        if t.purpose == "leisure" and t.accepted_recommendation
    )
    utilities = [
        leisure_net_utility(t)
        for a in sim.agents
        for t in a.trips
        if t.purpose == "leisure"
    ]
    record = {
        "leisure_trips": leisure,
        "acceptance_rate": (accepted / leisure) if leisure else 0.0,
        "mean_utility": float(np.mean(utilities)) if utilities else 0.0,
    }
    record.update(footfall_metrics(sim))
    record.update(taste_metrics(sim))
    return record


class RunMetrics:
    """Accumulates the by-segment panels over **every** day of a run.

    The paper's Table 1 and Table 2 are eval-day quantities and stay that way.
    The dashboard panels are not, and must not be: one evaluation day of 300
    agents yields around 40 leisure trips, and splitting 40 trips four ways by
    income and again four ways by trust leaves single-digit cells that are
    noise. Pooling the run multiplies the sample by the day count at no extra
    simulation cost.

    Only sums and counts are kept, never running means — merging means across
    days would silently weight a quiet day the same as a busy one.
    """

    def __init__(self, sim):
        # Segments are model *inputs* (income, trust, paradigm), so they are
        # fixed for the whole run and resolved once.
        self.segments = agent_segments(sim)
        self.nearest = _NearestPOI(sim)
        self.per_day: List[Dict[str, float]] = []
        self.welfare: Dict[str, Dict[str, Dict[str, float]]] = {
            "income": {}, "trust": {}, "paradigm": {}
        }
        self.spatial: Dict[str, Dict[str, object]] = {}
        self.category: Dict[str, Dict[str, int]] = {}
        self.taste_pool = {"all": [0, 0], "recommended": [0, 0], "organic": [0, 0]}
        # agent_id -> [sum of leisure net utility, trips], over the whole run.
        # The spatial welfare map pairs these against the control's, so it must
        # not be an eval-day quantity: on any one day only a small minority of
        # agents go out at all, and the intersection of "went out under the
        # control" and "went out under this arm" collapsed to a couple of dozen
        # points — far too sparse to bin into hexagons.
        self.agent_utility: Dict[int, List[float]] = {}

    @staticmethod
    def _spatial_bucket() -> Dict[str, object]:
        return {"n": 0, "distance_km": 0.0, "emissions_g": 0.0,
                "travel_min": 0.0, "detour": [], "excess_km": []}

    def observe_day(self, sim) -> None:
        self.per_day.append(day_metrics(sim))
        overall = self.spatial.setdefault("__overall__", self._spatial_bucket())

        for agent in sim.agents:
            groups = self.segments.get(agent.id)
            if groups is None:
                continue
            band = groups["income"]
            bucket = self.spatial.setdefault(band, self._spatial_bucket())
            row = self.category.setdefault(band, {})

            for trip in agent.trips:
                if trip.purpose != "leisure":
                    continue

                utility = leisure_net_utility(trip)
                pooled = self.agent_utility.setdefault(agent.id, [0.0, 0.0])
                pooled[0] += utility
                pooled[1] += 1.0

                for key, group in groups.items():
                    cell = self.welfare[key].setdefault(
                        group, {"n": 0, "sum_utility": 0.0, "n_negative": 0}
                    )
                    cell["n"] += 1
                    cell["sum_utility"] += utility
                    cell["n_negative"] += 1 if utility < 0 else 0

                for sink in (overall, bucket):
                    sink["n"] += 1
                    sink["distance_km"] += float(trip.distance_km)
                    sink["emissions_g"] += float(trip.emissions_g)
                    sink["travel_min"] += float(trip.travel_time_min)

                if trip.place_id:
                    place = sim.place_by_id.get(trip.place_id)
                    if place is not None:
                        row[place.category] = row.get(place.category, 0) + 1
                    matched = 1 if sim.taste_bonus(agent, trip.place_id) > 0.0 else 0
                    lane = "recommended" if trip.accepted_recommendation else "organic"
                    for key in ("all", lane):
                        self.taste_pool[key][0] += matched
                        self.taste_pool[key][1] += 1

                if trip.purpose_subtype:
                    floor_km = self.nearest.distance_km(
                        trip.purpose_subtype, trip.origin
                    )
                    # Sub-100 m floors make the ratio explode for an agent who
                    # happens to work next door to a candidate. Those trips say
                    # nothing about detour, so they are dropped rather than
                    # allowed to set the headline.
                    if floor_km is not None and floor_km >= 0.1:
                        chosen_km = float(
                            _haversine_vec(
                                trip.origin,
                                np.array([trip.destination[0]]),
                                np.array([trip.destination[1]]),
                            )[0]
                        )
                        ratio = chosen_km / floor_km
                        excess = chosen_km - floor_km
                        for sink in (overall, bucket):
                            sink["detour"].append(ratio)
                            sink["excess_km"].append(excess)

    @staticmethod
    def _finish_spatial(bucket: Dict[str, object]) -> Dict[str, float]:
        n = max(1, int(bucket["n"]))
        detour, excess = bucket["detour"], bucket["excess_km"]
        return {
            "n": int(bucket["n"]),
            "mean_distance_km": float(bucket["distance_km"]) / n,
            "mean_emissions_g": float(bucket["emissions_g"]) / n,
            "mean_travel_min": float(bucket["travel_min"]) / n,
            # Medians, not means: both are right-skewed, and one agent sent
            # across town would otherwise set the headline.
            #
            # The ratio ranks conditions correctly but is awkward to read on its
            # own, because in a dense core the nearest candidate of a subtype is
            # often 100-200 m away — so an ordinary 1 km outing already scores 5x
            # or more. `median_excess_km` is the same comparison in kilometres,
            # which is what the ratio should be quoted alongside.
            "median_detour": float(np.median(detour)) if detour else 0.0,
            "median_excess_km": float(np.median(excess)) if excess else 0.0,
        }

    def finalize(self, sim) -> Dict[str, object]:
        welfare = {
            key: {
                group: {
                    "n": int(cell["n"]),
                    "mean_utility": cell["sum_utility"] / max(1, cell["n"]),
                    "neg_rate": cell["n_negative"] / max(1, cell["n"]),
                }
                for group, cell in by_group.items()
            }
            for key, by_group in self.welfare.items()
        }
        return {
            "per_day": self.per_day,
            "run_utility": {
                int(agent_id): total / count
                for agent_id, (total, count) in self.agent_utility.items()
                if count > 0
            },
            "segments": welfare,
            "category_by_income": self.category,
            "spatial": {
                "overall": self._finish_spatial(
                    self.spatial.get("__overall__", self._spatial_bucket())
                ),
                "by_income": {
                    band: self._finish_spatial(bucket)
                    for band, bucket in self.spatial.items()
                    if band != "__overall__"
                },
            },
            "lorenz": lorenz_points(sim),
            "footfall": footfall_metrics(sim),
            "taste": {
                f"{key}_rate": (hits / total if total else 0.0)
                for key, (hits, total) in self.taste_pool.items()
            } | {
                f"{key}_n": total for key, (_h, total) in self.taste_pool.items()
            },
        }


def agent_frame(sim) -> Dict[str, list]:
    """Per-agent home location and segment, for the spatial welfare map.

    Identical across conditions at a given seed (homes and personas are inputs),
    so the study only needs this back from one arm. Returned as parallel lists
    rather than dicts: it is the one payload that scales with population.
    """
    segments = agent_segments(sim)
    ids, lats, lons, incomes, trust = [], [], [], [], []
    for agent in sim.agents:
        ids.append(int(agent.id))
        lats.append(round(float(agent.home[0]), 6))
        lons.append(round(float(agent.home[1]), 6))
        incomes.append(segments[agent.id]["income"])
        trust.append(segments[agent.id]["trust"])
    return {
        "agent_id": ids, "home_lat": lats, "home_lon": lons,
        "income": incomes, "trust": trust,
    }

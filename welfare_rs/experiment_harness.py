"""Paired-comparison experiment harness and welfare metrics.

Implements:
- Paired-comparison harness: runs matched-seed simulations with different
  RS treatments and computes per-agent welfare differences
- Gini coefficient of trip utility
- ORC (Over-Recommendation Cost) metric
- RM epsilon calibration from empirical regret distribution
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


# ── Welfare metrics ──────────────────────────────────────────────────────────

def gini_coefficient(values: Sequence[float]) -> float:
    """Compute the Gini coefficient of a distribution of values.

    Gini = 0 means perfect equality; Gini = 1 means perfect inequality.
    Handles negative values by shifting to non-negative domain.

    Parameters
    ----------
    values : Sequence[float]
        Utility values (one per agent or trip).

    Returns
    -------
    float
        Gini coefficient in [0, 1].
    """
    arr = np.array(values, dtype=float)
    if len(arr) < 2:
        return 0.0

    # Shift to non-negative if needed (standard approach for utility Gini)
    if arr.min() < 0:
        arr = arr - arr.min()

    # Avoid division by zero
    if arr.sum() == 0:
        return 0.0

    arr = np.sort(arr)
    n = len(arr)
    index = np.arange(1, n + 1)
    return float((2.0 * np.sum(index * arr) - (n + 1) * np.sum(arr)) / (n * np.sum(arr)))


def over_recommendation_cost(
    utility_with_rs: Sequence[float],
    utility_without_rs: Sequence[float],
) -> float:
    """Compute Over-Recommendation Cost (ORC).

    ORC = (1/N) Σ max(0, U_i^{no_rs} - U_i^{rs})

    Measures the average welfare loss from RS recommendations that made
    agents worse off than they would have been without any recommendation.

    Parameters
    ----------
    utility_with_rs : Sequence[float]
        Per-agent total utility under RS treatment.
    utility_without_rs : Sequence[float]
        Per-agent total utility under no-RS baseline.

    Returns
    -------
    float
        Average over-recommendation cost (non-negative).
    """
    u_rs = np.array(utility_with_rs, dtype=float)
    u_no = np.array(utility_without_rs, dtype=float)
    assert len(u_rs) == len(u_no), "Agent counts must match"
    losses = np.maximum(0.0, u_no - u_rs)
    return float(np.mean(losses))


def welfare_gain(
    utility_treatment: Sequence[float],
    utility_baseline: Sequence[float],
) -> Dict[str, float]:
    """Compute welfare comparison metrics between treatment and baseline.

    Returns
    -------
    Dict with keys:
        mean_gain: average per-agent utility gain
        median_gain: median per-agent utility gain
        pct_improved: fraction of agents with positive gain
        pct_harmed: fraction of agents with negative gain
        gini_treatment: Gini of treatment utilities
        gini_baseline: Gini of baseline utilities
        orc: Over-Recommendation Cost
    """
    u_t = np.array(utility_treatment, dtype=float)
    u_b = np.array(utility_baseline, dtype=float)
    diff = u_t - u_b

    return {
        "mean_gain": float(np.mean(diff)),
        "median_gain": float(np.median(diff)),
        "pct_improved": float(np.mean(diff > 0)),
        "pct_harmed": float(np.mean(diff < 0)),
        "gini_treatment": gini_coefficient(u_t),
        "gini_baseline": gini_coefficient(u_b),
        "orc": over_recommendation_cost(u_t, u_b),
    }


# ── Paper-exact Table 1 / Table 2 metrics ────────────────────────────────────
#
# These reproduce the metrics in the paper's Table 1 (aggregate welfare) and
# Table 2 (over-recommendation cost) *exactly*, over the leisure trips on the
# evaluation day. They are deliberately distinct from over_recommendation_cost()
# / welfare_gain() above, which average a clipped loss over *all* agents on total
# (all-trip) utility — a different quantity from the paper's per-harmed-agent ORC.

def leisure_net_utility(trip) -> float:
    """Paper-aligned realised net trip utility ``U = V - C`` for a leisure trip.

    ``V`` is the activity benefit (``trip.activity_utility``) and ``C`` is the
    generalized travel cost (``trip.gen_cost``) converted to utility units by the
    same ``gen_cost_denominator`` the simulation uses when travel cost enters mode
    utility — so V and C live on one scale, matching the paper's Eq. 1 minus the
    idiosyncratic term.
    """
    from . import params

    denom = params.UTILITY_WEIGHTS.get("gen_cost_denominator", 8.0) or 1.0
    return float(trip.activity_utility - trip.gen_cost / denom)


def eval_day_leisure_utilities(agents) -> Dict[int, float]:
    """Map ``agent_id -> mean leisure net utility`` over the trips currently held
    on each agent. After ``Simulation.run_days``, these are the last (evaluation)
    day's trips, since day planning clears ``agent.trips``."""
    out: Dict[int, float] = {}
    for a in agents:
        us = [leisure_net_utility(t) for t in a.trips if t.purpose == "leisure"]
        if us:
            out[a.id] = float(np.mean(us))
    return out


def table1_metrics(sim) -> Dict[str, float]:
    """Paper Table 1 metrics for a single run, over the eval-day leisure trips.

    Returns ``mean_utility`` (Ū), ``neg_rate`` (fraction of trips with U<0),
    ``gini`` of the trip-utility distribution, ``abstention_rate`` (fraction of
    leisure recommendation opportunities the RS withheld), and counts. ``sigma_u``
    (σ_U) is a *cross-seed* quantity and is computed when aggregating seeds, not
    here.
    """
    utils: List[float] = []
    for a in sim.agents:
        utils.extend(leisure_net_utility(t) for t in a.trips if t.purpose == "leisure")
    n = len(utils)
    opp = sum(1 for a in sim.agents if getattr(a, "had_leisure_opportunity", False))
    abst = sum(
        1 for a in sim.agents
        if getattr(a, "had_leisure_opportunity", False) and getattr(a, "rs_abstained", False)
    )
    return {
        "mean_utility": float(np.mean(utils)) if utils else 0.0,
        "neg_rate": float(np.mean([u < 0 for u in utils])) if utils else 0.0,
        "gini": gini_coefficient(utils) if n >= 2 else 0.0,
        "abstention_rate": (abst / opp) if opp else 0.0,
        "n_leisure_trips": n,
        "n_opportunities": opp,
    }


def table2_from_utilities(
    u_rec: Dict[int, float], u_org: Dict[int, float]
) -> Dict[str, float]:
    """Paper Table 2 from two ``agent_id -> eval-day leisure utility`` maps.

    Split out of :func:`table2_metrics` so the paired comparison can be made
    from *serialized* per-agent utilities instead of two live Simulation
    objects. That is what lets a treatment and its matched No-RS baseline run in
    separate processes: each worker returns only its own utility map (a few MB
    even at 100k agents) and the parent does the pairing afterwards.
    """
    common = sorted(set(u_rec) & set(u_org))
    n = len(common)
    if n == 0:
        return {"harmed_pct": 0.0, "improved_pct": 0.0, "mean_orc": 0.0, "n_matched": 0}
    diffs = [u_rec[i] - u_org[i] for i in common]  # rec - organic
    harmed = [d for d in diffs if d < 0]
    improved = [d for d in diffs if d > 0]
    return {
        "harmed_pct": len(harmed) / n,
        "improved_pct": len(improved) / n,
        # U_organic - U_rec over harmed agents = -(rec - organic).
        "mean_orc": float(np.mean([-d for d in harmed])) if harmed else 0.0,
        "n_matched": n,
    }


def table2_metrics(rec_sim, organic_sim) -> Dict[str, float]:
    """Paper Table 2 (over-recommendation cost) for a treatment vs its matched
    No-RS counterfactual, on the eval day.

    Agents are matched by id over those who took a leisure trip in *both* runs.
    Harmed % / Improved %: fraction whose eval-day leisure net utility is lower /
    higher under the treatment than under No RS. Mean ORC: mean of
    ``U_organic - U_rec`` over harmed agents only (Eq. 14).
    """
    return table2_from_utilities(
        eval_day_leisure_utilities(rec_sim.agents),
        eval_day_leisure_utilities(organic_sim.agents),
    )


# ── RM epsilon calibration ───────────────────────────────────────────────────

def calibrate_rm_epsilon(
    regret_values: Sequence[float],
    percentile: float = 75.0,
) -> float:
    """Calibrate RM epsilon threshold from empirical regret distribution.

    Sets ε at a given percentile of the observed regret distribution
    under the Standard RS condition.

    Parameters
    ----------
    regret_values : Sequence[float]
        Observed regret values from Standard RS simulation.
    percentile : float
        Percentile to use for threshold (default 75th).

    Returns
    -------
    float
        Calibrated epsilon threshold.
    """
    arr = np.array(regret_values, dtype=float)
    if len(arr) == 0:
        return 0.3  # fallback default
    return float(np.percentile(arr, percentile))


# ── Paired-comparison harness ────────────────────────────────────────────────

@dataclass
class TreatmentConfig:
    """Configuration for a single experimental treatment."""

    name: str
    welfare_mode: str  # "standard", "pup", "rm", "pup_rm", "oracle", "no_rs"
    pup_alpha: float = 0.6
    rm_epsilon: float = 0.3
    description: str = ""


@dataclass
class TreatmentResult:
    """Results from a single treatment run."""

    config: TreatmentConfig
    per_agent_utility: Dict[int, float]  # agent_id → total utility
    per_agent_trips: Dict[int, int]  # agent_id → trip count
    per_trip_utility: List[float]  # all trip utilities
    per_trip_regret: List[float]  # per-trip regret (vs. oracle)
    summary: Dict[str, Any]  # simulation summary dict
    feedback_history: List[Any] = field(default_factory=list)


DEFAULT_TREATMENTS = [
    TreatmentConfig("no_rs", "no_rs", description="No recommender system"),
    TreatmentConfig("standard", "standard", description="Standard RS (no welfare filter)"),
    TreatmentConfig("pup_60", "pup", pup_alpha=0.6, description="PUP-constrained (α=0.6)"),
    TreatmentConfig("pup_80", "pup", pup_alpha=0.8, description="PUP-constrained (α=0.8)"),
    TreatmentConfig("rm_30", "rm", rm_epsilon=0.3, description="RM-constrained (ε=0.3)"),
    TreatmentConfig("pup_rm", "pup_rm", pup_alpha=0.6, rm_epsilon=0.3, description="PUP+RM combined"),
    TreatmentConfig("oracle", "oracle", description="Oracle RS (perfect knowledge)"),
]


class ExperimentHarness:
    """Runs paired-comparison experiments across treatment conditions.

    All treatments use the same random seed and initial agent population
    to ensure matched comparisons. Welfare differences are computed
    per-agent across conditions.
    """

    def __init__(
        self,
        base_seed: int = 42,
        num_agents: int = 200,
        num_days: int = 5,
        treatments: Optional[List[TreatmentConfig]] = None,
    ):
        self.base_seed = base_seed
        self.num_agents = num_agents
        self.num_days = num_days
        self.treatments = treatments or DEFAULT_TREATMENTS
        self.results: Dict[str, TreatmentResult] = {}

    def run_treatment(self, config: TreatmentConfig) -> TreatmentResult:
        """Run a single treatment condition.

        This is a stub that documents the interface. The actual
        implementation requires importing and configuring Simulation,
        which creates circular dependencies. See the demonstration
        notebook for the full integration.
        """
        # NOTE: Full integration requires:
        # 1. Create Simulation with matched seed
        # 2. Configure WelfareAwareOrchestrator with treatment mode
        # 3. For "oracle" mode, register all agents with OracleRecommender
        # 4. For "no_rs" mode, disable recommender_stack
        # 5. Run simulation for num_days
        # 6. Collect per-agent and per-trip utilities

        raise NotImplementedError(
            f"run_treatment('{config.name}') requires Simulation integration. "
            "See experiment_demo.ipynb for the full workflow."
        )

    def run_all(self) -> Dict[str, TreatmentResult]:
        """Run all treatment conditions with matched seeds."""
        for config in self.treatments:
            self.results[config.name] = self.run_treatment(config)
        return self.results

    def compare(
        self,
        treatment_name: str,
        baseline_name: str = "no_rs",
    ) -> Dict[str, float]:
        """Compare a treatment against a baseline.

        Parameters
        ----------
        treatment_name : str
            Name of the treatment to evaluate.
        baseline_name : str
            Name of the baseline condition (default: "no_rs").

        Returns
        -------
        Dict[str, float]
            Welfare comparison metrics.
        """
        treatment = self.results[treatment_name]
        baseline = self.results[baseline_name]

        # Align agents by ID
        common_ids = sorted(
            set(treatment.per_agent_utility.keys()) & set(baseline.per_agent_utility.keys())
        )
        u_t = [treatment.per_agent_utility[aid] for aid in common_ids]
        u_b = [baseline.per_agent_utility[aid] for aid in common_ids]

        return welfare_gain(u_t, u_b)

    def compare_all(self, baseline_name: str = "no_rs") -> Dict[str, Dict[str, float]]:
        """Compare all treatments against a baseline."""
        comparisons = {}
        for name in self.results:
            if name == baseline_name:
                continue
            comparisons[name] = self.compare(name, baseline_name)
        return comparisons

    def calibrate_epsilon(self, standard_name: str = "standard") -> float:
        """Calibrate RM epsilon from the Standard RS run.

        Uses the 75th percentile of per-trip regret under Standard RS
        as the threshold for RM-constrained treatments.
        """
        if standard_name not in self.results:
            raise ValueError(f"Run '{standard_name}' treatment first")
        regrets = self.results[standard_name].per_trip_regret
        return calibrate_rm_epsilon(regrets)

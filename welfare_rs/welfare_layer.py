"""Utility-based welfare layer for the recommender system stack.

Implements:
- Continuous feedback signal (extends binary like/dislike → satisfaction + travel cost)
- Travel cost estimation from feedback history
- PUP (Positive Utility Probability) gating
- RM (Regret Minimization) gating
- WelfareAwareOrchestrator: wraps LeisureRSOrchestrator with welfare constraints
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from . import params
from .recommender_systems import (
    LeisureRSOrchestrator,
    Place,
    Recommendation,
    UserContext,
    euclidean_distance_km,
)


DEFAULT_SCORE_TO_UTILITY_MODE = "subtype_base_plus_score"
LEGACY_SCORE_TO_UTILITY_MODE = "identity"
DEFAULT_SCORE_UTILITY_SCALE = max(
    seg.get("activity_utility", 0.0)
    for seg in params.LEISURE_SEGMENTS.values()
)


# ── Continuous feedback ──────────────────────────────────────────────────────

@dataclass
class ContinuousFeedback:
    """Extended feedback signal beyond binary like/dislike.

    Attributes:
        user_id: Agent identifier.
        place_id: Place that was visited.
        source: Recommender source name.
        satisfaction: Continuous satisfaction score in [-1, 1].
        realized_travel_cost: Actual travel cost experienced (generalized cost).
        activity_utility: Realized activity utility.
        travel_utility: Realized travel utility (negative = costly).
        net_utility: satisfaction proxy = activity_utility + travel_utility.
    """

    user_id: int
    place_id: str
    source: str
    satisfaction: float  # [-1, 1]
    realized_travel_cost: float  # generalized cost in $-equivalent
    activity_utility: float
    travel_utility: float
    net_utility: float


def compute_continuous_feedback(
    trip,
    p_like: float,
    feedback_sensitivity: float = 1.0,
) -> ContinuousFeedback:
    """Convert a Trip + binary feedback signal into a continuous feedback record.

    The satisfaction score maps from the sigmoid probability p_like
    into a continuous [-1, 1] range, preserving the agent's nuanced
    evaluation rather than collapsing to binary.

    Parameters
    ----------
    trip : Trip
        The completed trip with utility components.
    p_like : float
        The probability of liking (from agent's feedback evaluation).
    feedback_sensitivity : float
        Agent-specific sensitivity multiplier.

    Returns
    -------
    ContinuousFeedback
        Continuous feedback record.
    """
    # Map p_like ∈ [0, 1] → satisfaction ∈ [-1, 1]
    satisfaction = (2.0 * p_like - 1.0) * min(2.0, max(0.2, feedback_sensitivity))
    satisfaction = max(-1.0, min(1.0, satisfaction))

    # Generalized travel cost: time cost + monetary cost
    # (travel_utility is already negative for costly trips)
    realized_travel_cost = trip.cost + (trip.travel_time_min / 60.0) * 15.0  # $15/hr default VOT

    net_utility = trip.activity_utility + trip.travel_utility

    return ContinuousFeedback(
        user_id=trip.agent_id,
        place_id=trip.recommended_place_id,
        source=trip.recommendation_source,
        satisfaction=satisfaction,
        realized_travel_cost=realized_travel_cost,
        activity_utility=trip.activity_utility,
        travel_utility=trip.travel_utility,
        net_utility=net_utility,
    )


# ── Travel cost estimation ───────────────────────────────────────────────────

class TravelCostEstimator:
    """Learns per-user travel cost parameters from feedback history.

    Maintains running estimates of:
    - β̂_i^time: user's marginal disutility of travel time ($/min)
    - β̂_i^cost: user's marginal disutility of monetary cost ($/$ spent)
    - v̂_i: user's revealed value of time ($/hr)

    Uses exponential moving average over realized trip costs.
    """

    def __init__(self, default_vot: float = 15.0, learning_rate: float = 0.15):
        self.default_vot = default_vot
        self.learning_rate = learning_rate
        # Per-user state: {user_id: {"vot": float, "avg_travel_cost": float, "n": int}}
        self.user_estimates: Dict[int, Dict[str, float]] = {}

    def update(self, feedback: ContinuousFeedback, travel_time_min: float, monetary_cost: float) -> None:
        """Update user travel cost estimates from a feedback observation."""
        state = self.user_estimates.setdefault(
            feedback.user_id,
            {"vot": self.default_vot, "avg_travel_cost": 0.0, "n": 0},
        )
        state["n"] += 1
        alpha = self.learning_rate

        # Update average travel cost with EMA
        realized_gc = monetary_cost + (travel_time_min / 60.0) * state["vot"]
        state["avg_travel_cost"] = (1 - alpha) * state["avg_travel_cost"] + alpha * realized_gc

        # Update VOT estimate: if user dislikes high-time trips, infer higher VOT
        # Simple heuristic: satisfaction < 0 on high-time trips → increase VOT
        if travel_time_min > 0 and feedback.satisfaction < -0.2:
            # User disliked a costly trip → they value time more
            state["vot"] = min(50.0, state["vot"] * (1 + 0.05 * abs(feedback.satisfaction)))
        elif travel_time_min > 20 and feedback.satisfaction > 0.3:
            # User liked a long trip → they don't mind travel as much
            state["vot"] = max(5.0, state["vot"] * (1 - 0.03 * feedback.satisfaction))

    def estimate_travel_cost(
        self,
        user_id: int,
        distance_km: float,
        avg_speed_kmh: float = 20.0,
        cost_per_km: float = 0.15,
    ) -> float:
        """Estimate generalized travel cost for a user-place pair.

        Returns Ĉ^travel_{ik} = β̂_i^time · τ̂ + β̂_i^cost · κ̂
        """
        state = self.user_estimates.get(user_id)
        if state is None:
            vot = self.default_vot
        else:
            vot = state["vot"]

        travel_time_hr = distance_km / max(1.0, avg_speed_kmh)
        monetary_cost = distance_km * cost_per_km
        return vot * travel_time_hr + monetary_cost

    def get_user_vot(self, user_id: int) -> float:
        """Return estimated VOT for a user."""
        state = self.user_estimates.get(user_id)
        return state["vot"] if state else self.default_vot

    def get_user_uncertainty(self, user_id: int) -> float:
        """Return uncertainty (σ) for user's utility estimate.

        Decreases with more observations. Used in PUP computation.
        """
        state = self.user_estimates.get(user_id)
        if state is None or state["n"] == 0:
            return 1.0  # high uncertainty
        # Uncertainty decays as 1/sqrt(n), floored at 0.15
        return max(0.15, 1.0 / math.sqrt(state["n"]))


# ── PUP and RM computation ───────────────────────────────────────────────────

def _normal_cdf(x: float) -> float:
    """Approximate standard normal CDF using the error function."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def compute_pup(
    v_hat_act: float,
    c_hat_travel: float,
    sigma_i: float,
) -> float:
    """Compute Positive Utility Probability.

    PUP = Pr[U_{ik} ≥ 0] ≈ Φ((V̂^act - Ĉ^travel) / σ_i)

    Parameters
    ----------
    v_hat_act : float
        Estimated activity benefit.
    c_hat_travel : float
        Estimated travel cost.
    sigma_i : float
        User-specific uncertainty parameter.

    Returns
    -------
    float
        Probability in [0, 1] that net utility is non-negative.
    """
    if sigma_i <= 0:
        return 1.0 if v_hat_act >= c_hat_travel else 0.0
    z = (v_hat_act - c_hat_travel) / sigma_i
    return _normal_cdf(z)


def compute_expected_regret(
    v_hat_act: float,
    c_hat_travel: float,
    sigma_i: float,
    best_known_utility: float = 0.0,
) -> float:
    """Compute expected regret for a recommendation.

    E[R_{ik}] = E[max(0, U* - U_{ik})]

    For a normal approximation:
    E[R] = σ · φ(z) + (best - μ) · Φ(-z)
    where μ = V̂^act - Ĉ^travel, z = (μ - best) / σ

    Parameters
    ----------
    v_hat_act : float
        Estimated activity benefit.
    c_hat_travel : float
        Estimated travel cost.
    sigma_i : float
        User-specific uncertainty.
    best_known_utility : float
        Best alternative utility (0 = doing nothing).

    Returns
    -------
    float
        Expected regret, non-negative.
    """
    mu = v_hat_act - c_hat_travel
    if sigma_i <= 1e-8:
        return max(0.0, best_known_utility - mu)

    z = (mu - best_known_utility) / sigma_i
    # φ(z) = standard normal pdf
    phi_z = math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    # Φ(-z)
    cdf_neg_z = _normal_cdf(-z)
    return sigma_i * phi_z + (best_known_utility - mu) * cdf_neg_z


def estimate_activity_utility_from_score(
    score: float,
    leisure_subtype: str,
    mode: str = DEFAULT_SCORE_TO_UTILITY_MODE,
    score_scale: float = DEFAULT_SCORE_UTILITY_SCALE,
) -> float:
    """Map an RS score into the welfare layer's activity-utility scale.

    Modes
    -----
    - ``identity``: legacy behavior, v_hat_act = score
    - ``subtype_base_plus_score``: add the subtype's assumed base activity
      utility to the RS score so the welfare layer operates on the same
      utility scale used elsewhere in the ABM.

    Set ``mode='identity'`` to revert to the previous behavior.
    """
    bounded_score = max(0.0, float(score))
    if mode == LEGACY_SCORE_TO_UTILITY_MODE:
        return bounded_score
    scaled_score = score_scale * bounded_score
    if mode == DEFAULT_SCORE_TO_UTILITY_MODE:
        base_utility = params.LEISURE_SEGMENTS.get(leisure_subtype, {}).get("activity_utility", 0.0)
        return base_utility + scaled_score
    raise ValueError(f"Unknown score-to-utility mode: {mode}")


# ── Welfare-aware orchestrator ───────────────────────────────────────────────

@dataclass
class WelfareAnnotatedRecommendation:
    """A recommendation annotated with welfare metrics."""

    recommendation: Recommendation
    v_hat_act: float
    c_hat_travel: float
    pup: float
    expected_regret: float
    passed_pup: bool
    passed_rm: bool


class WelfareAwareOrchestrator:
    """Wraps LeisureRSOrchestrator with PUP/RM gating.

    Treatment modes:
    - "standard": pass-through (no welfare filtering)
    - "pup": filter recommendations where PUP < α
    - "rm": filter recommendations where E[R] > ε
    - "pup_rm": apply both constraints
    """

    def __init__(
        self,
        base_orchestrator: LeisureRSOrchestrator,
        travel_cost_estimator: TravelCostEstimator,
        mode: str = "standard",
        pup_alpha: float = 0.6,
        rm_epsilon: float = 0.3,
        coord_scale_km: float = 1.0,
        coord_distance_km=euclidean_distance_km,
        score_to_utility_mode: str = DEFAULT_SCORE_TO_UTILITY_MODE,
        score_utility_scale: float = DEFAULT_SCORE_UTILITY_SCALE,
    ):
        self.base = base_orchestrator
        self.tce = travel_cost_estimator
        self.mode = mode
        self.pup_alpha = pup_alpha
        self.rm_epsilon = rm_epsilon
        self.coord_scale_km = coord_scale_km
        # Euclidean (grid) by default; OSM mode injects haversine for (lat, lon).
        self.coord_distance_km = coord_distance_km
        self.score_to_utility_mode = score_to_utility_mode
        self.score_utility_scale = score_utility_scale
        # Feedback history for continuous learning
        self.feedback_history: List[ContinuousFeedback] = []

    def recommend(
        self,
        user: UserContext,
        leisure_subtype: str,
        top_k_per_system: int = 5,
    ) -> Dict[str, List[WelfareAnnotatedRecommendation]]:
        """Get welfare-annotated recommendations.

        Returns recommendations from the base orchestrator, each annotated
        with PUP and RM scores, and filtered according to the active mode.
        """
        base_recs = self.base.recommend(user, leisure_subtype, top_k_per_system)
        sigma_i = self.tce.get_user_uncertainty(user.user_id)

        annotated: Dict[str, List[WelfareAnnotatedRecommendation]] = {}
        for source, recs in base_recs.items():
            annotated_list = []
            for rec in recs:
                # Estimate V̂^act on the same scale as ABM activity utility.
                v_hat_act = estimate_activity_utility_from_score(
                    rec.score,
                    leisure_subtype,
                    mode=self.score_to_utility_mode,
                    score_scale=self.score_utility_scale,
                )

                # Estimate Ĉ^travel
                dist_km = self.coord_distance_km(user.location, rec.place.location) * self.coord_scale_km
                c_hat_travel = self.tce.estimate_travel_cost(user.user_id, dist_km)

                pup = compute_pup(v_hat_act, c_hat_travel, sigma_i)
                e_regret = compute_expected_regret(v_hat_act, c_hat_travel, sigma_i)

                passed_pup = pup >= self.pup_alpha
                passed_rm = e_regret <= self.rm_epsilon

                annotated_list.append(
                    WelfareAnnotatedRecommendation(
                        recommendation=rec,
                        v_hat_act=v_hat_act,
                        c_hat_travel=c_hat_travel,
                        pup=pup,
                        expected_regret=e_regret,
                        passed_pup=passed_pup,
                        passed_rm=passed_rm,
                    )
                )

            # Apply welfare filter
            if self.mode == "pup":
                annotated_list = [a for a in annotated_list if a.passed_pup]
            elif self.mode == "rm":
                annotated_list = [a for a in annotated_list if a.passed_rm]
            elif self.mode == "pup_rm":
                annotated_list = [a for a in annotated_list if a.passed_pup and a.passed_rm]
            # "standard" → no filtering

            annotated[source] = annotated_list

        return annotated

    def record_continuous_feedback(self, feedback: ContinuousFeedback, travel_time_min: float, monetary_cost: float) -> None:
        """Record continuous feedback and update all learning components."""
        self.feedback_history.append(feedback)

        # Update travel cost estimator
        self.tce.update(feedback, travel_time_min, monetary_cost)

    def record_feedback(self, user_id: int, source: str, place_id: str, liked: bool, feedback_strength: float = 1.0) -> None:
        """Backward-compatible binary feedback passthrough."""
        self.base.record_feedback(
            user_id=user_id,
            source=source,
            place_id=place_id,
            liked=liked,
            feedback_strength=feedback_strength,
        )

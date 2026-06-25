"""Oracle recommender system — perfect-knowledge upper bound.

The Oracle RS has access to the agent's true utility function and always
recommends the activity that maximizes the agent's net utility (V^act - C^travel).
This serves as an upper bound on what any recommender could achieve.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from .recommender_systems import (
    LEISURE_SUBTYPE_TO_CATEGORIES,
    Place,
    Recommendation,
    RecommenderSystem,
    UserContext,
    euclidean_distance_km,
)


@dataclass
class AgentTruePreferences:
    """True preference parameters for an agent (known only to Oracle).

    In a real system these would be unobservable. The Oracle uses them
    to compute the true utility for each candidate place.

    Attributes:
        user_id: Agent identifier.
        vot: Value of time ($/hr).
        pref_time: Time sensitivity weight.
        pref_cost: Cost sensitivity weight.
        motivation_weights: Dict of motivation type → weight.
        leisure_preferences: Dict of subtype → preference multiplier.
        subtype_utility_bonuses: Dict of subtype → extra utility terms.
    """

    user_id: int
    vot: float
    pref_time: float
    pref_cost: float
    motivation_weights: Dict[str, float]
    leisure_preferences: Dict[str, float]
    subtype_utility_bonuses: Dict[str, float]


class OracleRecommender(RecommenderSystem):
    """Perfect-knowledge recommender that maximizes true agent utility.

    Given access to agent preferences (via register_agent), it computes
    the exact net utility U = V^act - C^travel for every candidate and
    returns the optimal ranking.
    """

    def __init__(
        self,
        catalog: Sequence[Place],
        coord_scale_km: float = 1.0,
        default_speed_kmh: float = 20.0,
        cost_per_km: float = 0.15,
    ):
        super().__init__(name="oracle", catalog=catalog)
        self.coord_scale_km = coord_scale_km
        self.default_speed_kmh = default_speed_kmh
        self.cost_per_km = cost_per_km
        self.agent_prefs: Dict[int, AgentTruePreferences] = {}

    def register_agent(self, prefs: AgentTruePreferences) -> None:
        """Register an agent's true preferences (simulation-only)."""
        self.agent_prefs[prefs.user_id] = prefs

    def _true_activity_utility(
        self,
        place: Place,
        prefs: AgentTruePreferences,
        leisure_subtype: str,
    ) -> float:
        """Compute true V^act for a place given true preferences."""
        # Base rating signal
        base = place.rating / 5.0

        # Preference fit for the subtype
        pref_multiplier = prefs.leisure_preferences.get(leisure_subtype, 1.0)

        # Subtype-specific bonus (from agent personality/motivation)
        subtype_bonus = prefs.subtype_utility_bonuses.get(leisure_subtype, 0.0)

        # Intrinsic motivation contribution
        intrinsic = prefs.motivation_weights.get("intrinsic", 0.25) * 0.3

        return base * pref_multiplier + subtype_bonus + intrinsic

    def _true_travel_cost(
        self,
        place: Place,
        user_location: Tuple[float, float],
        prefs: AgentTruePreferences,
    ) -> float:
        """Compute true C^travel for reaching a place."""
        dist_km = euclidean_distance_km(user_location, place.location) * self.coord_scale_km
        travel_time_hr = dist_km / max(1.0, self.default_speed_kmh)
        monetary_cost = dist_km * self.cost_per_km

        # C^travel = β^time · τ · v_i + β^cost · κ
        return (
            prefs.pref_time * travel_time_hr * prefs.vot
            + prefs.pref_cost * monetary_cost
        )

    def recommend(
        self,
        user: UserContext,
        leisure_subtype: str,
        top_k: int = 5,
    ) -> List[Recommendation]:
        """Return recommendations ranked by true net utility."""
        prefs = self.agent_prefs.get(user.user_id)
        if prefs is None:
            # Fallback: no oracle knowledge, return by rating
            candidates = self._filter_by_subtype(leisure_subtype)
            candidates.sort(key=lambda p: p.rating, reverse=True)
            return [
                Recommendation(
                    source=self.name,
                    place=p,
                    score=p.rating / 5.0,
                    components={"rating": p.rating / 5.0},
                )
                for p in candidates[:top_k]
            ]

        candidates = self._filter_by_subtype(leisure_subtype)
        if not candidates:
            return []

        scored = []
        for place in candidates:
            v_act = self._true_activity_utility(place, prefs, leisure_subtype)
            c_travel = self._true_travel_cost(place, user.location, prefs)
            net_utility = v_act - c_travel

            scored.append(
                Recommendation(
                    source=self.name,
                    place=place,
                    score=net_utility,
                    components={
                        "v_act": v_act,
                        "c_travel": c_travel,
                        "net_utility": net_utility,
                    },
                )
            )

        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[: max(1, top_k)]


class OracleOrchestrator:
    """Wrap OracleRecommender with the LeisureRSOrchestrator interface."""

    def __init__(self, oracle_rs: OracleRecommender):
        self.oracle = oracle_rs

    def recommend(
        self,
        user: UserContext,
        leisure_subtype: str,
        top_k_per_system: int = 5,
    ) -> Dict[str, List[Recommendation]]:
        return {
            self.oracle.name: self.oracle.recommend(
                user,
                leisure_subtype,
                top_k=top_k_per_system,
            )
        }

    def record_feedback(
        self,
        user_id: int,
        source: str,
        place_id: str,
        liked: bool,
        feedback_strength: float = 1.0,
    ) -> None:
        del user_id, source, place_id, liked, feedback_strength

    def record_continuous_feedback(self, feedback, travel_time_min: float, monetary_cost: float) -> None:
        del feedback, travel_time_min, monetary_cost


def register_agent_from_abm(oracle: OracleRecommender, agent) -> None:
    """Convenience: extract true preferences from an ABM Agent object."""
    from . import params

    subtype_bonuses = {}
    for subtype, seg_cfg in params.LEISURE_SEGMENTS.items():
        bonus = seg_cfg.get("activity_utility", 0.0)
        bonus += params.UTILITY_WEIGHTS["activity_intrinsic_coeff"] * agent.motivation_weights.get("intrinsic", 0.25)
        bonus += params.UTILITY_WEIGHTS["activity_escape_coeff"] * agent.motivation_weights.get("escape", 0.2)
        subtype_bonuses[subtype] = bonus

    prefs = AgentTruePreferences(
        user_id=agent.id,
        vot=getattr(agent, "vot", 15.0),
        pref_time=agent.preferences.get("time", 1.0),
        pref_cost=agent.preferences.get("cost", 1.0),
        motivation_weights=dict(agent.motivation_weights),
        leisure_preferences=dict(agent.leisure_preferences),
        subtype_utility_bonuses=subtype_bonuses,
    )
    oracle.register_agent(prefs)

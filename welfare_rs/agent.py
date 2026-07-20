"""Agent (traveller) class for the travel-behaviour ABM."""

from __future__ import annotations

import random
from collections.abc import Mapping

import numpy as np

from . import params
from .datastructures import Activity
from .utils import clamp, sigmoid, softmax
from .recommender_systems import LEISURE_SUBTYPE_TO_DEFAULT_KEYWORDS, UserContext


class Agent:
    """A heterogeneous traveler with preferences, motivations, and paradigms.

    This class operationalizes:
    - Personal characteristics (P): socio-demographics and attitudes.
    - Motivations: derived demand, intrinsic utility, escape, positionality.
    - Decision paradigms: utility, regret, prospect, satisficing, habit.
    - Self-determination theory: intrinsic vs extrinsic motivation split.
    """

    def __init__(self, agent_id, income, age, car_ownership, home, work, seed=0, eta_shift=0.0):
        ad = params.AGENT_DEFAULTS
        self.id = agent_id
        self.characteristics = {
            "income": income,
            "age": age,
            "car_ownership": car_ownership,
        }

        # Value of time (VOT).
        self.vot = max(ad["vot_floor"], income / ad["vot_hours_per_year"])

        rng = random.Random(seed + agent_id)
        self.preferences = {
            "time": rng.uniform(*ad["pref_time_range"]),
            "cost": rng.uniform(*ad["pref_cost_range"]),
            "comfort": rng.uniform(*ad["pref_comfort_range"]),
            "green": rng.uniform(*ad["pref_green_range"]),
        }

        # Attitudes (TPB) and social practice conformity.
        self.attitudes = {
            "travel_affinity": rng.uniform(*ad["attitude_range"]),
            "status_seeking": rng.uniform(*ad["attitude_range"]),
            "practice_conformity": rng.uniform(*ad["attitude_range"]),
        }

        # Motivation weights (sum to 1).
        weights = np.random.default_rng(seed + agent_id).dirichlet(ad["motivation_dirichlet"])
        self.motivation_weights = {
            "derived": weights[0],
            "intrinsic": weights[1],
            "escape": weights[2],
            "positionality": weights[3],
        }
        # TOGGLE: derived-demand-only ablation. Force all travel to be
        # classified as derived demand (zero out intrinsic/escape/positionality).
        if params.SIMPLIFICATION_TOGGLES.get("DERIVED_DEMAND_ONLY", False):
            self.motivation_weights = {
                "derived": 1.0,
                "intrinsic": 0.0,
                "escape": 0.0,
                "positionality": 0.0,
            }
        self.leisure_preferences = {
            segment: rng.uniform(*ad["leisure_pref_range"]) for segment in params.LEISURE_SEGMENTS
        }
        self.eta_shift = eta_shift
        self.last_eta = 0.0
        self.accepted_recommendation_count = 0
        self.daily_recommendations = {}
        # Per-day welfare-metric flags (reflect the most recently planned day).
        # had_leisure_opportunity: agent committed to a leisure outing (a slot
        # where the RS could have issued a suggestion). rs_abstained: that outing
        # received no recommendation (RS withheld / no candidate survived filtering,
        # or there was no RS at all). Together they yield the paper's abstention
        # rate over recommendation opportunities.
        self.had_leisure_opportunity = False
        self.rs_abstained = False
        self.persona_id = ""
        self.car_access_type = "own car" if car_ownership else "no car"
        self.transit_access_level = 0.5
        self.walk_tolerance_min = 15
        self.mobility_needs = "none"
        self.primary_interest = "cultural"
        self.time_window_pref = "weekday evening"
        self.feedback_sensitivity = rng.uniform(*ad["feedback_sensitivity_range"])
        self.car_access_penalty = 0.0

        # Survey/ICLV latent state (normalized to [0, 1]).
        self.big_five = {
            "openness": rng.uniform(*ad["psych_default_range"]),
            "conscientiousness": rng.uniform(*ad["psych_default_range"]),
            "extraversion": rng.uniform(*ad["psych_default_range"]),
            "agreeableness": rng.uniform(*ad["psych_default_range"]),
            "neuroticism": rng.uniform(*ad["psych_default_range"]),
        }
        self.latent_variables = {
            "maximization": rng.uniform(*ad["psych_default_range"]),
            "trust_platforms": rng.uniform(*ad["psych_default_range"]),
            "autonomy_preference": rng.uniform(*ad["psych_default_range"]),
            "algorithmic_awareness": rng.uniform(*ad["psych_default_range"]),
        }
        self.survey_items = {
            "local_leisure_frequency": 0.5,
            "overnight_trip_frequency": 0.5,
            "spontaneity_share": 0.5,
            "follow_through_friend": 0.5,
            "follow_through_platform": 0.5,
            "follow_through_ai": 0.5,
            "incentive_coupon": 0.5,
            "incentive_sponsored": 0.5,
            "incentive_loyalty": 0.5,
            "cross_platform_search": 0.5,
            "price_filter_tendency": 0.5,
            "group_coordination_preference": 0.5,
            "popularity_herding": 0.5,
            "unexpected_discovery": 0.5,
            "negative_recommendation_experience": 0.5,
            "review_posting": 0.5,
            "social_posting": 0.5,
            "switch_when_dissatisfied": 0.5,
            "multi_platform_parallel": 0.5,
            "advice_goal_directed": 0.5,
            "objective_algorithmic_literacy": 0.5,
            "city_familiarity": 0.5,
            "budget_tightness": 0.5,
            "ai_itinerary_comfort": 0.5,
            "explanation_needed": 0.5,
        }
        self.behavioral_coefficients = {}
        self.mode_preference_bias = {"walk": 0.0, "bike": 0.0, "transit": 0.0, "car": 0.0}

        # Decision-making paradigms (heterogeneous, can differ by choice type).
        pw = ad["paradigm_weights"]
        paradigm_choices = list(pw.keys())
        paradigm_wts = [pw[k] for k in paradigm_choices]
        self.decision_paradigms = {
            "mode_choice": rng.choices(paradigm_choices, weights=paradigm_wts, k=1)[0]
        }

        # Prospect theory parameters.
        self.loss_aversion = rng.uniform(*ad["loss_aversion_range"])
        self.reference_points = {
            "time_min": rng.randint(*ad["ref_time_range"]),
            "cost": rng.uniform(*ad["ref_cost_range"]),
        }

        # Satisficing threshold (generalized cost), scaled by income.
        self.satisficing_threshold = (
            rng.uniform(*ad["satisficing_base_range"]) + income / ad["satisficing_income_divisor"]
        )

        # Habit: default commute mode (updated after trips).
        self.habit_mode = "car" if car_ownership else "transit"

        self.home = home
        self.work = work
        self.schedule = []
        self.current_activity_index = 0
        self.activity_end_time = 0
        self.in_transit = False
        self.arrival_time = None
        self.total_delay = 0
        self.trips = []
        self.current_trip = None

        # Baselines are kept so survey updates can be applied repeatedly.
        self._base_preferences = dict(self.preferences)
        self._base_attitudes = dict(self.attitudes)
        self._base_motivation_weights = dict(self.motivation_weights)
        self._base_feedback_sensitivity = self.feedback_sensitivity
        self._base_loss_aversion = self.loss_aversion
        self._base_satisficing_threshold = self.satisficing_threshold
        self._refresh_behavior_from_survey()

    # ── Survey trait mapping ─────────────────────────────────────────────────

    @staticmethod
    def _mean(values):
        values = tuple(values)
        if not values:
            return 0.5
        return sum(values) / len(values)

    def sync_behavior_baselines(self):
        """Sync baseline utility primitives before applying survey overlays."""
        self._base_preferences = dict(self.preferences)
        self._base_attitudes = dict(self.attitudes)
        self._base_motivation_weights = dict(self.motivation_weights)
        self._base_feedback_sensitivity = self.feedback_sensitivity
        self._base_loss_aversion = self.loss_aversion
        self._base_satisficing_threshold = self.satisficing_threshold

    def _normalize_score(self, value, scale_min=1.0, scale_max=7.0):
        """Normalize a raw survey measurement into [0, 1]."""
        if value is None:
            return None
        try:
            score = float(value)
        except (TypeError, ValueError):
            return None
        if 0.0 <= score <= 1.0:
            return score
        if scale_max <= scale_min:
            return None
        return clamp((score - scale_min) / (scale_max - scale_min), 0.0, 1.0)

    def _normalized_or(self, mapping, key, fallback, scale_min=1.0, scale_max=7.0):
        value = mapping.get(key)
        normalized = self._normalize_score(value, scale_min=scale_min, scale_max=scale_max)
        if normalized is None:
            return fallback
        return normalized

    def _set_score_if_present(self, target, mapping, key, scale_min=1.0, scale_max=7.0):
        if key not in mapping:
            return
        normalized = self._normalize_score(mapping.get(key), scale_min=scale_min, scale_max=scale_max)
        if normalized is not None:
            target[key] = normalized

    def _refresh_behavior_from_survey(self):
        """Map latent variables and item scores to operational behavior."""
        sp = params.SURVEY_BEHAVIOR_PARAMS
        bf = self.big_five
        lv = self.latent_variables
        it = self.survey_items

        platform_affinity = clamp(
            sp["platform_affinity"]["trust"] * lv["trust_platforms"]
            + sp["platform_affinity"]["awareness"] * lv["algorithmic_awareness"]
            + sp["platform_affinity"]["platform_follow"] * it["follow_through_platform"]
            + sp["platform_affinity"]["ai_follow"] * it["follow_through_ai"],
            0.0,
            1.0,
        )
        autonomy_guard = clamp(
            sp["autonomy_guard"]["autonomy"] * lv["autonomy_preference"]
            + sp["autonomy_guard"]["awareness"] * lv["algorithmic_awareness"],
            0.0,
            1.0,
        )
        planning_orientation = clamp(
            sp["planning_orientation"]["conscientiousness"] * bf["conscientiousness"]
            + sp["planning_orientation"]["maximization"] * lv["maximization"]
            + sp["planning_orientation"]["low_spontaneity"] * (1.0 - it["spontaneity_share"]),
            0.0,
            1.0,
        )
        social_orientation = clamp(
            sp["social_orientation"]["extraversion"] * bf["extraversion"]
            + sp["social_orientation"]["group_coordination"] * it["group_coordination_preference"]
            + sp["social_orientation"]["social_posting"] * it["social_posting"],
            0.0,
            1.0,
        )
        variety_seeking = clamp(
            sp["variety_seeking"]["openness"] * bf["openness"]
            + sp["variety_seeking"]["unexpected_discovery"] * it["unexpected_discovery"]
            + sp["variety_seeking"]["advice_goal_directed"] * it["advice_goal_directed"],
            0.0,
            1.0,
        )
        risk_aversion = clamp(
            sp["risk_aversion"]["neuroticism"] * bf["neuroticism"]
            + sp["risk_aversion"]["autonomy"] * lv["autonomy_preference"]
            + sp["risk_aversion"]["awareness"] * lv["algorithmic_awareness"]
            + sp["risk_aversion"]["negative_experience"] * it["negative_recommendation_experience"],
            0.0,
            1.0,
        )
        budget_sensitivity = clamp(
            sp["budget_sensitivity"]["budget_tightness"] * it["budget_tightness"]
            + sp["budget_sensitivity"]["incentives"]
            * self._mean((it["incentive_coupon"], it["incentive_sponsored"], it["incentive_loyalty"])),
            0.0,
            1.0,
        )
        trend_susceptibility = clamp(
            sp["trend_susceptibility"]["herding"] * it["popularity_herding"]
            + sp["trend_susceptibility"]["agreeableness"] * bf["agreeableness"]
            + sp["trend_susceptibility"]["platform_follow"] * it["follow_through_platform"],
            0.0,
            1.0,
        )
        ai_affinity = clamp(
            sp["ai_affinity"]["ai_follow"] * it["follow_through_ai"]
            + sp["ai_affinity"]["ai_itinerary_comfort"] * it["ai_itinerary_comfort"]
            + sp["ai_affinity"]["trust"] * lv["trust_platforms"]
            + sp["ai_affinity"]["low_autonomy"] * (1.0 - lv["autonomy_preference"]),
            0.0,
            1.0,
        )
        feedback_loop_strength = clamp(
            sp["feedback_loop"]["review_posting"] * it["review_posting"]
            + sp["feedback_loop"]["switch_when_dissatisfied"] * it["switch_when_dissatisfied"]
            + sp["feedback_loop"]["multi_platform_parallel"] * it["multi_platform_parallel"],
            0.0,
            1.0,
        )

        self.behavioral_coefficients = {
            "platform_affinity": platform_affinity,
            "autonomy_guard": autonomy_guard,
            "planning_orientation": planning_orientation,
            "social_orientation": social_orientation,
            "variety_seeking": variety_seeking,
            "risk_aversion": risk_aversion,
            "budget_sensitivity": budget_sensitivity,
            "trend_susceptibility": trend_susceptibility,
            "ai_affinity": ai_affinity,
            "feedback_loop_strength": feedback_loop_strength,
        }

        # Remap core utility weights and attitudes.
        self.preferences = dict(self._base_preferences)
        self.attitudes = dict(self._base_attitudes)
        self.motivation_weights = dict(self._base_motivation_weights)
        self.feedback_sensitivity = self._base_feedback_sensitivity
        self.loss_aversion = self._base_loss_aversion
        self.satisficing_threshold = self._base_satisficing_threshold

        self.preferences["time"] = clamp(
            self.preferences["time"]
            + sp["preference_shift"]["time_planning"] * (planning_orientation - 0.5)
            + sp["preference_shift"]["time_maximization"] * (lv["maximization"] - 0.5),
            sp["preference_bounds"]["min"],
            sp["preference_bounds"]["max"],
        )
        self.preferences["cost"] = clamp(
            self.preferences["cost"]
            + sp["preference_shift"]["cost_budget"] * (budget_sensitivity - 0.5)
            + sp["preference_shift"]["cost_price_filter"] * (it["price_filter_tendency"] - 0.5),
            sp["preference_bounds"]["min"],
            sp["preference_bounds"]["max"],
        )
        self.preferences["comfort"] = clamp(
            self.preferences["comfort"] + sp["preference_shift"]["comfort_risk"] * (risk_aversion - 0.5),
            0.0,
            1.5,
        )
        self.preferences["green"] = clamp(
            self.preferences["green"] + sp["preference_shift"]["green_openness"] * (bf["openness"] - 0.5),
            0.0,
            1.2,
        )

        self.attitudes["travel_affinity"] = clamp(
            self.attitudes["travel_affinity"]
            + sp["attitude_shift"]["travel_affinity_variety"] * (variety_seeking - 0.5)
            + sp["attitude_shift"]["travel_affinity_frequency"] * (it["local_leisure_frequency"] - 0.5),
            0.0,
            1.0,
        )
        self.attitudes["status_seeking"] = clamp(
            self.attitudes["status_seeking"] + sp["attitude_shift"]["status_trend"] * (trend_susceptibility - 0.5),
            0.0,
            1.0,
        )
        self.attitudes["practice_conformity"] = clamp(
            self.attitudes["practice_conformity"]
            + sp["attitude_shift"]["practice_social"] * (social_orientation - 0.5)
            + sp["attitude_shift"]["practice_trend"] * (trend_susceptibility - 0.5),
            0.0,
            1.0,
        )
        # TOGGLE: when DERIVED_DEMAND_ONLY is on, skip all motivation updates
        # and keep weights pinned at pure-derived-demand.
        if params.SIMPLIFICATION_TOGGLES.get("DERIVED_DEMAND_ONLY", False):
            self.motivation_weights = {
                "derived": 1.0,
                "intrinsic": 0.0,
                "escape": 0.0,
                "positionality": 0.0,
            }
        else:
            mw = dict(self.motivation_weights)
            mw["derived"] += sp["motivation_shift"]["derived_planning"] * (planning_orientation - 0.5)
            mw["intrinsic"] += sp["motivation_shift"]["intrinsic_variety"] * (variety_seeking - 0.5)
            mw["escape"] += sp["motivation_shift"]["escape_spontaneity"] * (it["spontaneity_share"] - 0.5)
            mw["positionality"] += sp["motivation_shift"]["positionality_trend"] * (trend_susceptibility - 0.5)
            total = sum(max(0.01, v) for v in mw.values())
            for key in mw:
                mw[key] = max(0.01, mw[key]) / total
            self.motivation_weights = mw

        self.feedback_sensitivity = clamp(
            self.feedback_sensitivity
            + sp["feedback_sensitivity_shift"]["feedback_loop"] * (feedback_loop_strength - 0.5)
            + sp["feedback_sensitivity_shift"]["risk_aversion"] * (risk_aversion - 0.5),
            0.5,
            1.8,
        )
        self.loss_aversion = clamp(
            self.loss_aversion + sp["prospect_shift"]["loss_aversion_risk"] * (risk_aversion - 0.5),
            1.05,
            4.0,
        )
        self.satisficing_threshold = max(
            1.0,
            self.satisficing_threshold
            + sp["satisficing_shift"]["budget"] * (budget_sensitivity - 0.5)
            - sp["satisficing_shift"]["maximization"] * (lv["maximization"] - 0.5),
        )
        paradigm_scores = {
            "utility": 1.0 + sp["paradigm_shift"]["utility_variety"] * variety_seeking,
            "regret": 1.0 + sp["paradigm_shift"]["regret_maximization"] * lv["maximization"],
            "prospect": 1.0 + sp["paradigm_shift"]["prospect_risk"] * risk_aversion,
            "satisficing": (
                1.0
                + sp["paradigm_shift"]["satisficing_budget"] * budget_sensitivity
                + sp["paradigm_shift"]["satisficing_planning"] * planning_orientation
            ),
            "habit": (
                1.0
                + sp["paradigm_shift"]["habit_autonomy"] * autonomy_guard
                + sp["paradigm_shift"]["habit_low_variety"] * (1.0 - variety_seeking)
            ),
        }
        # Use paradigm scores as sampling weights so all five paradigms can emerge
        _p_names = list(paradigm_scores.keys())
        _p_weights = [paradigm_scores[k] for k in _p_names]
        _prng = random.Random(self.id + 9999)
        self.decision_paradigms["mode_choice"] = _prng.choices(_p_names, weights=_p_weights, k=1)[0]
        self.mode_preference_bias = {
            "walk": clamp(
                sp["mode_bias"]["walk_variety"] * (variety_seeking - 0.5)
                + sp["mode_bias"]["walk_budget"] * (budget_sensitivity - 0.5)
                - sp["mode_bias"]["walk_risk"] * (risk_aversion - 0.5),
                -0.4,
                0.4,
            ),
            "bike": clamp(
                sp["mode_bias"]["bike_variety"] * (variety_seeking - 0.5)
                + sp["mode_bias"]["bike_green"] * (self.preferences["green"] - 0.5)
                - sp["mode_bias"]["bike_risk"] * (risk_aversion - 0.5),
                -0.4,
                0.4,
            ),
            "transit": clamp(
                sp["mode_bias"]["transit_budget"] * (budget_sensitivity - 0.5)
                + sp["mode_bias"]["transit_platform"] * (platform_affinity - 0.5),
                -0.4,
                0.4,
            ),
            "car": clamp(
                sp["mode_bias"]["car_autonomy"] * (autonomy_guard - 0.5)
                + sp["mode_bias"]["car_risk"] * (risk_aversion - 0.5)
                - sp["mode_bias"]["car_green"] * (self.preferences["green"] - 0.5),
                -0.4,
                0.4,
            ),
        }

    def _subtype_personality_adjustment(self, subtype):
        """Adjust leisure subtype utility using latent and item-level signals."""
        sb = params.SURVEY_BEHAVIOR_PARAMS["subtype_utility_shift"]
        bc = self.behavioral_coefficients
        adjustment = 0.0
        if subtype in sb["social_subtypes"]:
            adjustment += sb["social_bonus"] * (bc["social_orientation"] - 0.5)
        if subtype in sb["exploration_subtypes"]:
            adjustment += sb["exploration_bonus"] * (bc["variety_seeking"] - 0.5)
        if subtype in sb["planned_subtypes"]:
            adjustment += sb["planned_bonus"] * (bc["planning_orientation"] - 0.5)
        if subtype in sb["spontaneous_subtypes"]:
            adjustment += sb["spontaneous_bonus"] * (self.survey_items["spontaneity_share"] - 0.5)
        if subtype in sb["outdoor_subtypes"]:
            adjustment -= sb["outdoor_risk_penalty"] * (bc["risk_aversion"] - 0.5)
        if subtype in sb["trend_subtypes"]:
            adjustment += sb["trend_bonus"] * (bc["trend_susceptibility"] - 0.5)
        return adjustment

    def _trip_disutility_multiplier(self):
        """Scale expected travel burden from psychological and survey factors."""
        dm = params.SURVEY_BEHAVIOR_PARAMS["trip_disutility_multiplier"]
        bc = self.behavioral_coefficients
        multiplier = (
            1.0
            + dm["risk_aversion"] * (bc["risk_aversion"] - 0.5)
            + dm["budget_sensitivity"] * (bc["budget_sensitivity"] - 0.5)
            - dm["variety_seeking"] * (bc["variety_seeking"] - 0.5)
            - dm["platform_affinity"] * (bc["platform_affinity"] - 0.5)
        )
        return clamp(multiplier, dm["min"], dm["max"])

    def apply_survey_profile(self, profile):
        """Apply measured latent variables and survey items to this agent.

        Expected schema (all keys optional):
            {
                "big_five": {...},
                "latent_variables": {...},
                "items": {...}
            }
        Values may be normalized [0,1] or raw Likert-style scores.
        """
        if not isinstance(profile, Mapping):
            return
        bf = profile.get("big_five", {})
        lv = profile.get("latent_variables", {})
        items = profile.get("items", {})
        if not isinstance(bf, Mapping):
            bf = {}
        if not isinstance(lv, Mapping):
            lv = {}
        if not isinstance(items, Mapping):
            items = {}

        for key in self.big_five:
            self._set_score_if_present(self.big_five, bf, key)
        for key in self.latent_variables:
            self._set_score_if_present(self.latent_variables, lv, key)

        # 7-point items by default.
        for key in (
            "follow_through_friend",
            "follow_through_platform",
            "follow_through_ai",
            "cross_platform_search",
            "price_filter_tendency",
            "group_coordination_preference",
            "popularity_herding",
            "unexpected_discovery",
            "negative_recommendation_experience",
            "review_posting",
            "social_posting",
            "switch_when_dissatisfied",
            "multi_platform_parallel",
            "ai_itinerary_comfort",
            "explanation_needed",
            "city_familiarity",
        ):
            self._set_score_if_present(self.survey_items, items, key)

        # 5-point items.
        for key in ("incentive_coupon", "incentive_sponsored", "incentive_loyalty"):
            self._set_score_if_present(self.survey_items, items, key, scale_min=1.0, scale_max=5.0)

        # Categorical/frequency items pre-scaled to [0, 1] or given in percentages/count bins.
        self.survey_items["spontaneity_share"] = self._normalized_or(
            items,
            "spontaneity_share",
            self.survey_items["spontaneity_share"],
            scale_min=0.0,
            scale_max=100.0,
        )
        self.survey_items["local_leisure_frequency"] = self._normalized_or(
            items,
            "local_leisure_frequency",
            self.survey_items["local_leisure_frequency"],
            scale_min=0.0,
            scale_max=7.0,
        )
        self.survey_items["overnight_trip_frequency"] = self._normalized_or(
            items,
            "overnight_trip_frequency",
            self.survey_items["overnight_trip_frequency"],
            scale_min=0.0,
            scale_max=12.0,
        )
        self.survey_items["budget_tightness"] = self._normalized_or(
            items,
            "budget_tightness",
            self.survey_items["budget_tightness"],
            scale_min=0.0,
            scale_max=1.0,
        )
        self.survey_items["advice_goal_directed"] = self._normalized_or(
            items,
            "advice_goal_directed",
            self.survey_items["advice_goal_directed"],
            scale_min=0.0,
            scale_max=1.0,
        )
        self.survey_items["objective_algorithmic_literacy"] = self._normalized_or(
            items,
            "objective_algorithmic_literacy",
            self.survey_items["objective_algorithmic_literacy"],
            scale_min=0.0,
            scale_max=1.0,
        )

        self._refresh_behavior_from_survey()

    # ── Leisure helpers ──────────────────────────────────────────────────────

    def _authority_window(self, env, segment):
        """Return opening window for a leisure subtype."""
        authority = env.context.get("authority_constraints", {})
        by_subtype = authority.get("leisure_hours_by_subtype", {})
        return by_subtype.get(segment, authority.get("leisure_open", (9 * 60, 23 * 60)))

    def _interest_keywords(self):
        """Build user interest keywords from top leisure preferences."""
        ranked = sorted(
            self.leisure_preferences.items(),
            key=lambda item: item[1],
            reverse=True,
        )
        top = [segment for segment, _ in ranked[:3]]
        keywords = []
        for segment in top:
            keywords.extend(LEISURE_SUBTYPE_TO_DEFAULT_KEYWORDS.get(segment, ()))
        seen = set()
        deduped = []
        for token in keywords:
            token_l = token.lower()
            if token_l in seen:
                continue
            seen.add(token_l)
            deduped.append(token_l)
        return tuple(deduped)

    # ── Eta (recommendation acceptance) ──────────────────────────────────────

    def _estimate_eta(self, recommendation_score):
        """Willingness to accept a recommendation (eta).

        eta = baseline(trust, autonomy) + population-wide eta_shift
              + quality term (RS score) + memory term (accepted count),
        clamped to [eta_min, eta_max]. See ``params.SIMPLIFIED_ETA_PARAMS``.
        """
        sep = params.SIMPLIFIED_ETA_PARAMS
        trust = self.latent_variables.get("trust_platforms", 0.5)
        autonomy = self.latent_variables.get("autonomy_preference", 0.5)
        eta_base = (
            sep["baseline_intercept"]
            + sep["baseline_trust_coeff"] * (trust - 0.5)
            - sep["baseline_autonomy_coeff"] * (autonomy - 0.5)
        )
        quality_effect = sep["quality_weight"] * (recommendation_score - 0.5)
        memory_effect = sep["memory_weight"] * min(
            sep["memory_cap"], self.accepted_recommendation_count
        )
        eta = eta_base + self.eta_shift + quality_effect + memory_effect
        return clamp(eta, sep["eta_min"], sep["eta_max"])

    # ── Feedback ─────────────────────────────────────────────────────────────

    def evaluate_recommendation_feedback(self, trip, env, rng):
        """Generate like/dislike feedback for accepted recommendations."""
        if trip.purpose != "leisure" or not trip.accepted_recommendation:
            return 0, 0.0

        fp = params.FEEDBACK_PARAMS
        sp = params.SURVEY_BEHAVIOR_PARAMS["feedback_shift"]
        subtype = trip.purpose_subtype
        pref_fit = self.leisure_preferences.get(subtype, 1.0) - 1.0
        intrinsic = self.motivation_weights["intrinsic"]
        escape = self.motivation_weights["escape"]
        travel_affinity = self.attitudes["travel_affinity"]

        context_term = 0.0
        weather = env.context.get("weather", "fair")
        if weather in {"rain", "heat"} and subtype in {"park", "workout_or_run"}:
            context_term += fp["rain_heat_outdoor_penalty"]
        if env.context.get("social_norms") == "green" and subtype == "park":
            context_term += fp["green_norm_park_bonus"]

        experience_term = fp["experience_activity_weight"] * trip.activity_utility + fp["experience_travel_weight"] * trip.travel_utility
        score = (
            experience_term
            + fp["pref_fit_weight"] * pref_fit
            + fp["intrinsic_weight"] * intrinsic
            + fp["escape_weight"] * escape
            + fp["travel_affinity_weight"] * travel_affinity
            + sp["trust_platforms"] * (self.latent_variables["trust_platforms"] - 0.5)
            + sp["feedback_loop_strength"] * (self.behavioral_coefficients["feedback_loop_strength"] - 0.5)
            - sp["autonomy_preference"] * (self.latent_variables["autonomy_preference"] - 0.5)
            - sp["awareness_caution"] * (self.latent_variables["algorithmic_awareness"] - 0.5)
            + context_term
        )
        score *= 1.0 + sp["feedback_sensitivity_multiplier"] * (self.feedback_sensitivity - 1.0)
        p_like = clamp(sigmoid(fp["sigmoid_scale"] * score), fp["p_like_min"], fp["p_like_max"])
        liked = 1 if rng.random() < p_like else 0
        return liked, p_like

    # ── TPB intention ────────────────────────────────────────────────────────

    def _tpb_intention(self, env, purpose):
        """Theory of Planned Behavior: intention from attitude, norm, PBC."""
        tp = params.TPB_PARAMS
        attitude = self.attitudes["travel_affinity"]
        if purpose == "leisure":
            attitude += tp["leisure_attitude_boost"] * self.motivation_weights["intrinsic"]

        norm = tp["norm_neutral"]
        if env.context.get("travel_norm") == "pro_travel":
            norm = tp["norm_pro_travel"]
        elif env.context.get("travel_norm") == "anti_travel":
            norm = tp["norm_anti_travel"]

        pbc = tp["pbc_base"]
        if self.car_access_type == "own car":
            pbc += tp["pbc_own_car"]
        elif self.car_access_type == "carshare":
            pbc += tp["pbc_carshare"]
        pbc += tp["pbc_transit_weight"] * (self.transit_access_level - 0.5)
        if self.characteristics["age"] > 75:
            pbc += tp["pbc_age_75_penalty"]
        if self.mobility_needs == "ADA/wheelchair":
            pbc += tp["pbc_ada_penalty"]
        pbc = clamp(pbc)

        return sigmoid(tp["attitude_coeff"] * attitude + tp["norm_coeff"] * norm + tp["pbc_coeff"] * pbc + tp["intercept"])

    # ── Day planning ─────────────────────────────────────────────────────────

    def plan_day(self, env, rng, recommender_stack=None, day_index=0):
        """Generate a daily activity schedule with RS-driven leisure choices.

        ``env`` is the owning :class:`welfare_rs.Simulation` — it provides the
        shared context (weather, norms, authority constraints), network
        distances, and POI sampling.
        """
        del day_index
        ad = params.AGENT_DEFAULTS
        pp = params.PARTICIPATION_PARAMS
        sp = params.SURVEY_BEHAVIOR_PARAMS["participation_shift"]
        ls = params.LEISURE_SEGMENTS

        base_remote = ad["remote_base"] + (ad["remote_no_car_boost"] if self.characteristics["car_ownership"] is False else 0.0)
        base_remote += ad["remote_age_55_boost"] if self.characteristics["age"] > 55 else 0.0
        remote_today = rng.random() < min(ad["remote_cap"], base_remote)

        work_start = rng.randint(*ad["work_start_range"])
        work_duration = rng.randint(*ad["work_duration_range"])

        schedule = []
        schedule.append(Activity("home", "", "organic", False, "", 0, work_start, self.home, is_mandatory=True))
        self.last_eta = 0.0
        self.had_leisure_opportunity = False
        self.rs_abstained = False

        if remote_today:
            schedule.append(Activity("work", "", "organic", False, "", work_start, work_duration, self.home, is_mandatory=True))
        else:
            schedule.append(Activity("work", "", "organic", False, "", work_start, work_duration, self.work, is_mandatory=True))

        after_work_start = work_start + work_duration
        origin_after_work = self.home if remote_today else self.work

        self.daily_recommendations = {}

        # (1)-(2) Evaluate leisure participation and subtype choice from net utility.
        # The RS is deliberately NOT consulted here: participation and subtype
        # choice are built from organic ingredients only, so the (expensive) RS
        # query can wait until a subtype is actually chosen — see step (3).
        intention = self._tpb_intention(env, "leisure")
        motivation_boost = (
            pp["motivation_intrinsic_coeff"] * self.motivation_weights["intrinsic"]
            + pp["motivation_escape_coeff"] * self.motivation_weights["escape"]
        )
        motivation_boost += sp["social_orientation"] * (self.behavioral_coefficients["social_orientation"] - 0.5)
        motivation_boost += sp["variety_seeking"] * (self.behavioral_coefficients["variety_seeking"] - 0.5)
        motivation_boost += sp["local_frequency"] * (self.survey_items["local_leisure_frequency"] - 0.5)

        options = []
        for subtype, seg_cfg in ls.items():
            leisure_delay = rng.randint(*seg_cfg["delay_minmax"])
            leisure_duration = rng.randint(*seg_cfg["duration_minmax"])
            leisure_start = after_work_start + leisure_delay
            # Organic leisure destination: a real POI of this subtype, chosen by
            # proximity (falls back to a random node if the catalog has none).
            desired_location, desired_place_id = env.sample_poi(subtype, origin_after_work, rng)

            open_start, open_end = self._authority_window(env, subtype)
            if leisure_start < open_start:
                leisure_start = open_start
            if leisure_start + leisure_duration > open_end:
                continue

            d_out = env.distance_km(origin_after_work, desired_location)
            d_back = env.distance_km(desired_location, self.home)
            expected_trip_disutility = pp["expected_trip_disutility_per_km"] * (d_out + d_back)
            expected_trip_disutility *= self._trip_disutility_multiplier()
            if subtype in {"park", "workout_or_run"} and self.walk_tolerance_min >= pp["outdoor_walk_tol_threshold"]:
                expected_trip_disutility *= pp["outdoor_walk_tol_discount"]

            activity_utility = seg_cfg["activity_utility"]
            activity_utility += pp["activity_intrinsic_coeff"] * self.motivation_weights["intrinsic"]
            activity_utility += pp["activity_escape_coeff"] * self.motivation_weights["escape"]
            activity_utility += pp["activity_travel_affinity_coeff"] * self.attitudes["travel_affinity"]
            activity_utility += pp["activity_pref_coeff"] * (self.leisure_preferences[subtype] - 1.0)
            activity_utility += self._subtype_personality_adjustment(subtype)

            interest_map = pp["interest_map"]
            if subtype in interest_map.get(self.primary_interest, set()):
                activity_utility += pp["primary_interest_bonus"]

            hour = leisure_start / 60.0
            tw_cfg = pp["time_window_bonuses"].get(self.time_window_pref)
            if tw_cfg is not None:
                lo, hi = tw_cfg["range"]
                if lo <= hour <= hi:
                    activity_utility += tw_cfg["in_range"]
                else:
                    activity_utility += tw_cfg["out_range"]

            net_utility = activity_utility - expected_trip_disutility + rng.uniform(*pp["net_utility_noise"])

            options.append(
                {
                    "subtype": subtype,
                    "start": leisure_start,
                    "duration": leisure_duration,
                    "desired_location": desired_location,
                    "desired_place_id": desired_place_id,
                    "activity_utility": activity_utility,
                    "net_utility": net_utility,
                }
            )

        chosen = None
        do_leisure = False
        if options:
            best_net = max(o["net_utility"] for o in options)
            participation_signal = (
                pp["signal_intercept"]
                + pp["signal_intention_coeff"] * intention
                + pp["signal_motivation_coeff"] * motivation_boost
                + pp["signal_net_utility_coeff"] * best_net
            )
            participation_signal += sp["spontaneity"] * (self.survey_items["spontaneity_share"] - 0.5)
            participation_signal -= sp["negative_experience"] * (self.survey_items["negative_recommendation_experience"] - 0.5)
            participation_signal += sp["city_familiarity"] * (self.survey_items["city_familiarity"] - 0.5)
            p_participate = clamp(sigmoid(participation_signal), 0.0, pp["p_participate_max"])
            do_leisure = (rng.random() < p_participate) and (best_net > pp["min_net_utility"])

            if do_leisure:
                temperature = pp["softmax_temperature"]
                temperature *= 1.0 + sp["softmax_maximization"] * (self.latent_variables["maximization"] - 0.5)
                temperature *= 1.0 - sp["softmax_spontaneity"] * (self.survey_items["spontaneity_share"] - 0.5)
                temperature = max(sp["softmax_min"], temperature)
                probs = softmax([o["net_utility"] * temperature for o in options])
                chosen = rng.choices(options, weights=probs, k=1)[0]

        # (3)-(4) Query the RS for the chosen subtype only, then decide
        # acceptance with dynamic eta. Recommenders are deterministic and
        # side-effect-free (no RNG, no state mutation), so querying after the
        # organic participation/subtype choice is behaviour-identical to the
        # old query-all-subtypes-up-front — agents who stay home skip the RS
        # entirely, and the other six subtypes are never scored.
        if do_leisure and chosen is not None:
            subtype = chosen["subtype"]
            best_rec = None
            if recommender_stack is not None:
                user_ctx = UserContext(
                    user_id=self.id,
                    location=origin_after_work,
                    query_keywords=LEISURE_SUBTYPE_TO_DEFAULT_KEYWORDS.get(subtype, ()),
                    interest_keywords=self._interest_keywords(),
                )
                by_source = recommender_stack.recommend(user_ctx, subtype, top_k_per_system=3)
                by_source = {
                    source: [getattr(rec, "recommendation", rec) for rec in recs]
                    for source, recs in by_source.items()
                }
                flat = [rec for recs in by_source.values() for rec in recs]
                best_rec = max(flat, key=lambda rec: rec.score) if flat else None
                self.daily_recommendations[subtype] = {"by_source": by_source, "best": best_rec}
            # Abstention bookkeeping: this leisure outing is a recommendation
            # opportunity; it counts as withheld when the RS surfaced no candidate
            # for the chosen subtype (filtered out under PUP/RM, none available, or
            # no RS at all). Mirrors Table 1's "Abst." column.
            self.had_leisure_opportunity = True
            self.rs_abstained = best_rec is None
            eta = 0.0
            accepted = False
            chosen_location = chosen["desired_location"]
            chosen_place_id = chosen["desired_place_id"]
            source = "organic"
            if best_rec is not None:
                eta = self._estimate_eta(best_rec.score)
                accepted = rng.random() < eta
                if accepted:
                    chosen_location = best_rec.place.location
                    chosen_place_id = best_rec.place.place_id
                    source = best_rec.source

            self.last_eta = eta
            if accepted:
                self.accepted_recommendation_count += 1

            schedule.append(
                Activity(
                    "leisure",
                    subtype,
                    source,
                    accepted,
                    chosen_place_id if accepted else "",
                    chosen["start"],
                    chosen["duration"],
                    chosen_location,
                    is_mandatory=False,
                    place_id=chosen_place_id,
                )
            )
            after_work_start = chosen["start"] + chosen["duration"]

        schedule.append(
            Activity(
                "home",
                "",
                "organic",
                False,
                "",
                after_work_start,
                max(10, 1440 - after_work_start),
                self.home,
                is_mandatory=True,
            )
        )

        self.schedule = schedule
        self.current_activity_index = 0
        self.activity_end_time = schedule[0].start_time + schedule[0].duration
        self.in_transit = False
        self.arrival_time = None
        self.total_delay = 0
        self.trips = []
        self.current_trip = None

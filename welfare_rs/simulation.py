"""Simulation orchestrator for the travel-behaviour ABM."""

from __future__ import annotations

import copy
import csv
import hashlib
import math
import random
from collections import Counter
from pathlib import Path

from . import params
from .agent import Agent
from .datastructures import Trip
from .utils import clamp, haversine_km, softmax
from .recommender_systems import (
    Place,
    PlaceDynamics,
    build_recommender_stack,
)


class SimulationCancelled(Exception):
    """Raised by :meth:`Simulation.run_days` when a ``should_stop`` callback
    signals cancellation between days. Picklable (no args) so it propagates
    cleanly out of worker processes."""


class Simulation:
    """Orchestrates the day, applies mode choice, and aggregates statistics.

    The simulation runs on a real OSM street network (:class:`welfare_rs.geo.
    RoadNetwork`, required); every location in the model is a (lat, lon) network
    node. The simulation itself is the agents' environment: it owns the shared
    context (weather, norms, authority constraints), network distances, the POI
    catalog, and the live per-POI visit/feedback state (:class:`PlaceDynamics`).
    """

    def __init__(
        self,
        num_agents=None,
        seed=None,
        time_step=None,
        context=None,
        use_recommenders=True,
        rs_policy=None,
        eta_shift=0.0,
        use_persona_agents=True,
        persona_csv_path=None,
        poi_csv_path=None,
        road_network=None,
        disabled_modes=None,
        recommender_override=None,
        recommender_factory=None,
    ):
        if road_network is None:
            raise ValueError(
                "Simulation requires a road_network (build one with "
                "welfare_rs.build_road_network(<city preset>))."
            )
        sd = params.SIM_DEFAULTS
        cp = params.CITY_PARAMS
        self.seed = seed if seed is not None else sd["seed"]
        self.rng = random.Random(self.seed)
        # Dedicated stream for home/work/fallback location sampling, so those
        # draws are independent of the main decision stream.
        self._loc_rng = random.Random(self.seed)
        self.time_step = time_step if time_step is not None else sd["time_step"]
        self.use_recommenders = use_recommenders
        self.rs_policy = rs_policy or {}
        self.eta_shift = eta_shift
        self.use_persona_agents = use_persona_agents
        self.persona_csv_path = persona_csv_path if persona_csv_path is not None else params.NYC_PERSONA_CSV_PATH
        self.poi_csv_path = poi_csv_path
        self.road_network = road_network

        # Shared context (weather, norms, authority constraints) — deep-copy so
        # runtime patches don't mutate the template in params.py.
        self.context = copy.deepcopy(cp["default_context"])
        self.context["community_mode_bias"] = dict(cp["community_mode_bias"])
        if context:
            self.context.update(context)

        # Congestion capacities scale with network size. Base (intersection)
        # nodes only, so capacity doesn't drift when POIs are inserted as extra
        # mid-block nodes or when a POI-augmented network is reused across runs.
        area = road_network.num_base_nodes
        self.road_capacity = max(
            cp["road_capacity_floor"], int(area * cp["road_capacity_multiplier"])
        )
        self.transit_capacity = max(
            cp["transit_capacity_floor"], int(area * cp["transit_capacity_multiplier"])
        )

        # Mode parameters — read from params at init time. ``disabled_modes``
        # lets a caller drop modes entirely (e.g., the frontend removes transit).
        self.modes = {k: dict(v) for k, v in params.MODE_PARAMS.items()}
        for _m in (disabled_modes or ()):
            self.modes.pop(_m, None)
        self.mode_status = dict(params.MODE_STATUS)
        self.mode_enjoyment = dict(params.MODE_ENJOYMENT)

        # Place catalog and agents must exist before some recommender factories
        # can finish wiring treatment-specific stacks (for example Oracle).
        self.place_catalog = self._build_place_catalog()

        # Live per-POI rating/review/popularity state, fed by visits and
        # like/dislike feedback during the run and consumed by the recommenders.
        self.place_dynamics = PlaceDynamics(self.place_catalog)

        # Index the catalog by leisure subtype so organic leisure choices resolve
        # to real POIs (sample_poi) rather than random intersections — both
        # organic and recommended visits land on catalog places, which is what
        # makes footfall-per-POI meaningful and matches the paper's proximity-based
        # self-selected alternative.
        from .recommender_systems import LEISURE_SUBTYPE_TO_CATEGORIES

        self._pois_by_subtype = {}
        for _subtype, _cats in LEISURE_SUBTYPE_TO_CATEGORIES.items():
            _catset = set(_cats)
            self._pois_by_subtype[_subtype] = [p for p in self.place_catalog if p.category in _catset]

        _num_agents = num_agents if num_agents is not None else sd["num_agents"]
        self.agents = []
        personas = self._load_personas(self.persona_csv_path) if self.use_persona_agents else []
        # The given personas are used as-is for the first len(personas) agents.
        # Beyond that, synthesise realistic, non-duplicate personas + homes (see
        # _synthesize_persona) so the population can exceed the provided set.
        used_homes = set()
        for p in personas:
            try:
                used_homes.add((round(float(p["start_latitude"]), 6), round(float(p["start_longitude"]), 6)))
            except (KeyError, ValueError, TypeError):
                pass
        syn_rng = random.Random(self.seed + 991)
        for i in range(_num_agents):
            if personas:
                persona = personas[i] if i < len(personas) else self._synthesize_persona(i, personas, syn_rng, used_homes)
                agent = self._build_agent_from_persona(i, persona)
            else:
                agent = self._build_random_agent(i)
            self.agents.append(agent)

        # Routing only ever runs between agent homes/works and catalog POIs;
        # registering them lets the network prune each cached Dijkstra tree to
        # just these target nodes (rare fallback destinations self-heal lazily).
        self.road_network.register_route_targets(
            [p.location for p in self.place_catalog]
            + [a.home for a in self.agents]
            + [a.work for a in self.agents]
        )

        self.recommender_stack = self._resolve_recommender_stack(
            recommender_override=recommender_override,
            recommender_factory=recommender_factory,
        )
        self._plan_agents_for_day(day_index=0)

        self.stats = None
        self.last_road_volume = 0
        self.last_transit_volume = 0
        self.last_mode_counts = Counter()

    # ── Environment interface (shared with Agent.plan_day) ───────────────────

    def distance_km(self, a, b):
        """Network shortest-path distance in km between two (lat, lon) nodes."""
        return self.road_network.route_length_km_latlon(a, b)

    def community_mode_bias(self, mode):
        """Return a social-practice bias for a mode (SPT)."""
        return self.context.get("community_mode_bias", {}).get(mode, 0.0)

    def sample_poi(self, subtype, origin, rng):
        """Sample an organic destination POI for a leisure subtype.

        Among the catalog POIs of the chosen subtype, a place is drawn with
        probability ``∝ exp(-dist/scale)`` so closer options dominate — the
        proximity-driven "self-selected alternative" from the paper. Distances
        use straight-line (haversine) proximity, mirroring the recommenders'
        proximity heuristic; realized travel cost is still measured on the
        network downstream.

        Returns ``(location, place_id)``; falls back to a random intersection
        (with ``place_id=""``) when the catalog has no POI for the subtype.
        """
        places = self._pois_by_subtype.get(subtype) or []
        if not places:
            return self.road_network.sample_node_latlon(self._loc_rng), ""
        if len(places) == 1:
            return places[0].location, places[0].place_id
        scale = params.CITY_PARAMS.get("organic_poi_proximity_scale_km", 3.0) or 1.0
        weights = [math.exp(-haversine_km(origin, p.location) / scale) for p in places]
        if sum(weights) <= 0:
            place = rng.choice(places)
        else:
            place = rng.choices(places, weights=weights, k=1)[0]
        return place.location, place.place_id

    # ── Persona loading ──────────────────────────────────────────────────────

    def _load_personas(self, persona_csv_path):
        if not persona_csv_path:
            return []
        path = Path(persona_csv_path)
        if not path.exists():
            return []
        rows = []
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("PersonaID"):
                    rows.append(row)
        return rows

    def _home_from_latlon(self, lat, lon):
        """Map a persona's real (lat, lon) to the nearest network node."""
        return self.road_network.snap_latlon(lat, lon)

    @staticmethod
    def _coerce_unit_interval(value):
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        token = str(value).strip().lower()
        if token == "":
            return None
        lookup = {
            "yes": 1.0,
            "no": 0.0,
            "true": 1.0,
            "false": 0.0,
            "high": 0.8,
            "medium": 0.5,
            "med": 0.5,
            "low": 0.2,
            "option a": 1.0,
            "a": 1.0,
            "option b": 0.0,
            "b": 0.0,
            "it depends": 0.5,
            "depends": 0.5,
        }
        if token in lookup:
            return lookup[token]
        try:
            return float(token)
        except ValueError:
            return None

    @staticmethod
    def _first_nonempty(row, keys):
        for key in keys:
            value = row.get(key)
            if value is None:
                continue
            if str(value).strip() == "":
                continue
            return value
        return None

    def _build_survey_profile_from_persona(self, persona):
        """Extract optional measured latent variables/items from persona rows."""
        if not isinstance(persona, dict):
            return {}

        profile = {"big_five": {}, "latent_variables": {}, "items": {}}
        big_five_alias = {
            "openness": ("openness", "bigfive_openness", "bfi_openness", "lv_openness", "Openness"),
            "conscientiousness": (
                "conscientiousness",
                "bigfive_conscientiousness",
                "bfi_conscientiousness",
                "lv_conscientiousness",
                "Conscientiousness",
            ),
            "extraversion": ("extraversion", "bigfive_extraversion", "bfi_extraversion", "lv_extraversion", "Extraversion"),
            "agreeableness": (
                "agreeableness",
                "bigfive_agreeableness",
                "bfi_agreeableness",
                "agreeable_composite",
                "Agreeableness",
            ),
            "neuroticism": ("neuroticism", "bigfive_neuroticism", "bfi_neuroticism", "neurotic_composite", "Neuroticism"),
        }
        latent_alias = {
            "maximization": ("maximization", "lv_maximization", "Maximization"),
            "trust_platforms": ("trust_platforms", "lv_trust_platforms", "platform_trust"),
            "autonomy_preference": ("autonomy_preference", "lv_autonomy", "autonomy_control"),
            "algorithmic_awareness": ("algorithmic_awareness", "lv_algorithmic_awareness", "algorithm_awareness"),
        }
        item_alias = {
            "local_leisure_frequency": ("q12_leisure_freq", "local_leisure_frequency"),
            "overnight_trip_frequency": ("q13_overnight_freq", "overnight_trip_frequency"),
            "spontaneity_share": ("q14_spontaneity", "spontaneity_share"),
            "follow_through_friend": ("q17_follow_friend", "follow_through_friend"),
            "follow_through_platform": ("q18_follow_platform", "follow_through_platform"),
            "follow_through_ai": ("q19_follow_ai", "follow_through_ai"),
            "incentive_coupon": ("q2_5a_coupon", "incentive_coupon"),
            "incentive_sponsored": ("q2_5b_sponsored", "incentive_sponsored"),
            "incentive_loyalty": ("q2_5c_loyalty", "incentive_loyalty"),
            "cross_platform_search": ("search_depth", "cross_platform_search"),
            "price_filter_tendency": ("price_filter_tendency", "price_filter_use"),
            "group_coordination_preference": ("group_coordination_preference",),
            "popularity_herding": ("q2_4i_trending", "popularity_herding"),
            "unexpected_discovery": ("q2_4e_serendipity", "unexpected_discovery"),
            "negative_recommendation_experience": (
                "q2_4k_negative_platform",
                "q6_1f_disappointing_recommendation",
                "negative_recommendation_experience",
            ),
            "review_posting": ("q6_4a_review", "review_posting"),
            "social_posting": ("q6_4b_social_post", "social_posting"),
            "switch_when_dissatisfied": ("q6_4c_switch_platform", "switch_when_dissatisfied"),
            "multi_platform_parallel": ("q6_4d_multi_platform", "multi_platform_parallel"),
            "advice_goal_directed": ("q22_advice_preference", "advice_goal_directed"),
            "objective_algorithmic_literacy": ("q21_algorithmic_literacy", "objective_algorithmic_literacy"),
            "city_familiarity": ("q11_city_familiarity", "city_familiarity"),
            "budget_tightness": ("budget_tightness",),
            "ai_itinerary_comfort": ("q6_1e_ai_itinerary", "ai_itinerary_comfort"),
            "explanation_needed": ("q6_1c_explanation_trust", "explanation_needed"),
        }

        for canonical, aliases in big_five_alias.items():
            raw = self._first_nonempty(persona, aliases)
            value = self._coerce_unit_interval(raw)
            if value is not None:
                profile["big_five"][canonical] = value

        for canonical, aliases in latent_alias.items():
            raw = self._first_nonempty(persona, aliases)
            value = self._coerce_unit_interval(raw)
            if value is not None:
                profile["latent_variables"][canonical] = value

        for canonical, aliases in item_alias.items():
            raw = self._first_nonempty(persona, aliases)
            value = self._coerce_unit_interval(raw)
            if value is not None:
                profile["items"][canonical] = value

        # Fall back to existing persona fields when direct survey columns are absent.
        if "budget_tightness" not in profile["items"]:
            budget = str(persona.get("Budget", "medium")).strip().lower()
            profile["items"]["budget_tightness"] = {"low": 0.85, "medium": 0.50, "high": 0.20}.get(budget, 0.50)
        if "follow_through_ai" not in profile["items"]:
            willingness_ai = str(persona.get("WillingnessAI", "med")).strip().lower()
            profile["items"]["follow_through_ai"] = {"high": 0.80, "med": 0.55, "low": 0.25}.get(willingness_ai, 0.55)
        if "follow_through_platform" not in profile["items"]:
            profile["items"]["follow_through_platform"] = profile["items"]["follow_through_ai"]
        if "popularity_herding" not in profile["items"]:
            top_rated = str(persona.get("TopRated", "no")).strip().lower() == "yes"
            profile["items"]["popularity_herding"] = 0.75 if top_rated else 0.40
        if "spontaneity_share" not in profile["items"]:
            group = str(persona.get("Group", "solo")).strip().lower()
            profile["items"]["spontaneity_share"] = 0.62 if "friends" in group else 0.45
        if "city_familiarity" not in profile["items"]:
            profile["items"]["city_familiarity"] = 0.65

        has_payload = any(profile[section] for section in ("big_five", "latent_variables", "items"))
        return profile if has_payload else {}

    # ── Agent builders ───────────────────────────────────────────────────────

    def _build_random_agent(self, i):
        ad = params.AGENT_DEFAULTS
        income = self.rng.randint(*ad["income_range"])
        age = self.rng.randint(*ad["age_range"])
        if income < ad["car_ownership_income_threshold"]:
            car_ownership = self.rng.random() < ad["car_ownership_prob_low_income"]
        else:
            car_ownership = self.rng.random() < ad["car_ownership_prob_high_income"]
        home = self.road_network.sample_node_latlon(self._loc_rng)
        work = self.road_network.sample_node_latlon(self._loc_rng)
        agent = Agent(i, income, age, car_ownership, home, work, seed=self.seed, eta_shift=self.eta_shift)
        agent.car_access_type = "own car" if car_ownership else "no car"
        agent.car_access_penalty = 0.0 if car_ownership else ad["no_car_access_penalty"]
        return agent

    def _build_agent_from_persona(self, i, persona):
        pm = params.PERSONA_MAPPING
        ad = params.AGENT_DEFAULTS

        income_band = str(persona.get("Income", "35-75k")).strip()
        age_band = str(persona.get("Age", "mid adult")).strip()
        inc_lo, inc_hi = pm["income_bands"].get(income_band, (35000, 74000))
        age_lo, age_hi = pm["age_bands"].get(age_band, (30, 55))
        income = self.rng.randint(inc_lo, inc_hi)
        age = self.rng.randint(age_lo, age_hi)

        car_access = str(persona.get("CarAccess", "no car")).strip()
        transit_access = str(persona.get("TransitAccess", "medium")).strip()
        mobility_needs = str(persona.get("MobilityNeeds", "none")).strip()
        env_conscious = str(persona.get("EnvConscious", "low")).strip()
        risk_salience = str(persona.get("RiskSalience", "low")).strip()
        walk_tol = int(float(str(persona.get("WalkTolerance", "15")).strip()))
        budget = str(persona.get("Budget", "medium")).strip()
        primary_interest = str(persona.get("PrimaryInterest", "cultural")).strip()
        time_window = str(persona.get("TimeWindow", "weekday evening")).strip()
        group = str(persona.get("Group", "solo")).strip()
        top_rated = str(persona.get("TopRated", "no")).strip().lower() == "yes"

        # TOGGLE: when CAR_ONLY_MODE is on we neutralise CarAccess /
        # TransitAccess persona columns. Every agent is treated as a car
        # owner with no access penalty, which prevents these persona
        # columns from leaking into the results.
        if params.SIMPLIFICATION_TOGGLES.get("CAR_ONLY_MODE", False):
            car_access = "own car"
            car_ownership = True
            car_access_penalty = 0.0
        elif car_access == "own car":
            car_ownership = True
            car_access_penalty = 0.0
        elif car_access == "carshare":
            car_ownership = False
            car_access_penalty = pm["carshare_penalty"]
        else:
            car_ownership = False
            car_access_penalty = pm["no_car_penalty"]

        try:
            lat = float(persona.get("start_latitude", ""))
            lon = float(persona.get("start_longitude", ""))
            home = self._home_from_latlon(lat, lon)
        except ValueError:
            home = self.road_network.sample_node_latlon(self._loc_rng)
        work = self.road_network.sample_node_latlon(self._loc_rng)

        agent = Agent(i, income, age, car_ownership, home, work, seed=self.seed, eta_shift=self.eta_shift)

        agent.persona_id = str(persona.get("PersonaID", f"P{i:04d}"))
        agent.car_access_type = car_access
        agent.car_access_penalty = car_access_penalty
        agent.transit_access_level = pm["transit_access_levels"].get(transit_access, 0.55)
        agent.walk_tolerance_min = max(5, min(45, walk_tol))
        agent.mobility_needs = mobility_needs
        agent.primary_interest = primary_interest
        agent.time_window_pref = time_window
        agent.feedback_sensitivity = 1.1 if risk_salience == "high" else 0.95

        agent.characteristics["car_ownership"] = car_ownership

        agent.preferences["green"] = 0.75 if env_conscious == "high" else 0.25
        if budget == "low":
            agent.preferences["cost"] = 1.2
        elif budget == "high":
            agent.preferences["cost"] = 0.85
        else:
            agent.preferences["cost"] = 1.0
        agent.preferences["time"] = 1.1 if time_window == "weekday evening" else 0.95
        agent.attitudes["travel_affinity"] = 0.65 if group in {"solo", "mixed friends 20-30"} else 0.5
        agent.attitudes["status_seeking"] = 0.7 if primary_interest == "nightlife" else 0.4
        agent.attitudes["practice_conformity"] = 0.7 if top_rated else 0.45

        # TOGGLE: DERIVED_DEMAND_ONLY skips all persona-driven motivation
        # adjustments and pins the weights to pure-derived-demand.
        if params.SIMPLIFICATION_TOGGLES.get("DERIVED_DEMAND_ONLY", False):
            agent.motivation_weights = {
                "derived": 1.0,
                "intrinsic": 0.0,
                "escape": 0.0,
                "positionality": 0.0,
            }
        else:
            ma = pm["motivation_adjustments"]
            mw = dict(agent.motivation_weights)
            if primary_interest in {"food", "cultural"}:
                mw["derived"] += ma["food_cultural_derived"]
            if primary_interest in {"nightlife", "nature"}:
                mw["intrinsic"] += ma["nightlife_nature_intrinsic"]
            if group == "solo":
                mw["escape"] += ma["solo_escape"]
            if top_rated:
                mw["positionality"] += ma["top_rated_positionality"]
            total = sum(max(0.01, v) for v in mw.values())
            for k in mw:
                mw[k] = max(0.01, mw[k]) / total
            agent.motivation_weights = mw

        agent.sync_behavior_baselines()

        # Survey overlay: measured latent variables / items from the persona row
        # (plus coarse fallbacks from stated columns like Budget / WillingnessAI)
        # remap the agent's behavioural coefficients.
        survey_profile = self._build_survey_profile_from_persona(persona)
        if survey_profile:
            agent.apply_survey_profile(survey_profile)

        return agent

    def _synthesize_persona(self, idx, personas, rng, used_homes):
        """Extrapolate a new persona to extend the population past the given set.

        Each attribute is resampled independently from the base personas' empirical
        distribution, so per-attribute frequencies match the provided population
        while the combination is new (not a duplicate row). The home is a fresh,
        distinct node from the active road network — a realistic on-street location,
        the same way the base homes were sampled. Caveat: independent resampling
        reproduces the marginals but not cross-attribute correlations.
        """
        columns = list(personas[0].keys())
        synthetic = {col: rng.choice([p.get(col, "") for p in personas]) for col in columns}
        synthetic["PersonaID"] = f"SYN{idx:04d}"

        home = self.road_network.sample_node_latlon(rng)
        for _ in range(25):  # resample to keep homes distinct
            if (round(home[0], 6), round(home[1], 6)) not in used_homes:
                break
            home = self.road_network.sample_node_latlon(rng)
        used_homes.add((round(home[0], 6), round(home[1], 6)))
        synthetic["start_latitude"] = f"{home[0]:.6f}"
        synthetic["start_longitude"] = f"{home[1]:.6f}"
        return synthetic

    # ── Place catalog ────────────────────────────────────────────────────────

    def _synth_prominence(self, place_id):
        """Deterministic synthetic rating / review_count / popularity for a place.

        The real POI dataset has no ratings or reviews, which the recommender
        prominence signal needs. We derive them deterministically from the place
        id (independent of the simulation seed) so they are stable across runs and
        identical across treatments — i.e. not a confound in matched comparisons.
        """
        cp = params.CATALOG_PARAMS
        h = int(hashlib.md5(str(place_id).encode("utf-8")).hexdigest()[:8], 16)
        r = random.Random(h)
        rating = round(r.uniform(*cp["rating_range"]), 2)
        review_count = r.randint(*cp["review_count_range"])
        popularity = max(cp["popularity_floor"], review_count * r.uniform(*cp["popularity_multiplier_range"]))
        return rating, review_count, popularity

    def _load_osm_poi_catalog(self, poi_csv_path):
        """Load real NYC leisure POIs (from data/filter_nyc_pois.py) onto the network.

        POIs are restricted to the active network's bounds and capped per category
        for responsiveness; rating/review/popularity are synthesised (the source
        lacks them — see ``_synth_prominence``). Returns ``[]`` when no POI file is
        available, so the simulation falls back to ``_build_synthetic_osm_catalog``.
        """
        if not poi_csv_path:
            return []
        path = Path(poi_csv_path)
        if not path.exists():
            return []

        south, west, north, east = self.road_network.bounds
        cap = params.POI_PARAMS.get("max_per_category", 600)
        by_category: dict = {}
        seen_ids = set()
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                try:
                    lat = float(row["latitude"])
                    lon = float(row["longitude"])
                except (KeyError, ValueError):
                    continue
                if not (south <= lat <= north and west <= lon <= east):
                    continue
                category = (row.get("category") or "").strip()
                if category not in params.CATEGORY_KEYWORDS:
                    continue
                place_id = row.get("place_id", "")
                if place_id and place_id in seen_ids:
                    continue
                seen_ids.add(place_id)
                by_category.setdefault(category, []).append(
                    (place_id, row.get("name", ""), category, lat, lon)
                )

        # Cap per category with a fixed seed (not the simulation seed) so the POI
        # set is identical across treatments/seeds — keeping the network mutation
        # idempotent and matched comparisons clean.
        cap_rng = random.Random(20240608)
        selected = []
        for rows in by_category.values():
            if cap and len(rows) > cap:
                rows = cap_rng.sample(rows, cap)
            selected.extend(rows)

        # Insert each POI as a mid-block graph node (at the projection of its real
        # coordinate onto the nearest street) so routing is granular within a
        # block. The POI keeps its EXACT (lat, lon) from the dataset as its
        # location — routing snaps that coordinate to its own inserted node, and
        # the map shows it at the true building position.
        self.road_network.add_pois_as_nodes(
            [(place_id, lat, lon) for (place_id, _n, _c, lat, lon) in selected]
        )

        catalog = []
        for place_id, name, cat, lat, lon in selected:
            rating, review_count, popularity = self._synth_prominence(place_id)
            catalog.append(
                Place(
                    place_id=place_id or f"poi_{len(catalog) + 1}",
                    name=name or cat,
                    category=cat,
                    location=(lat, lon),
                    keywords=tuple(params.CATEGORY_KEYWORDS.get(cat, (cat,))),
                    rating=rating,
                    review_count=review_count,
                    popularity=popularity,
                )
            )
        return catalog

    def _build_default_recommender_stack(self):
        # The RS proximity heuristic uses straight-line (haversine) distance —
        # the paper's "as the crow flies" signal, deliberately cruder than the
        # network cost of the realised trip (haversine already returns km, so
        # no coordinate scaling is needed).
        return build_recommender_stack(
            self.place_catalog,
            google_maps_config=self.rs_policy.get("google_maps", {}),
            popularity_config=self.rs_policy.get("popularity", {}),
            dynamics=self.place_dynamics,
        )

    def _resolve_recommender_stack(self, recommender_override=None, recommender_factory=None):
        if recommender_override is not None:
            return recommender_override
        base_stack = self._build_default_recommender_stack()
        if recommender_factory is not None:
            return recommender_factory(self, base_stack)
        return base_stack

    def _build_synthetic_osm_catalog(self):
        """Synthetic POIs placed on real network nodes.

        Fallback when no real POI dataset is available (see
        ``_load_osm_poi_catalog``). Every category in ``CATEGORY_KEYWORDS`` gets a
        handful of places, guaranteeing each leisure subtype has candidates.
        """
        cp = params.CATALOG_PARAMS
        lo, hi = cp["osm_places_per_category"]
        catalog = []
        place_index = 0
        for category, keywords in params.CATEGORY_KEYWORDS.items():
            for _ in range(self.rng.randint(lo, hi)):
                place_index += 1
                location = self.road_network.sample_node_latlon(self.rng)
                rating = round(self.rng.uniform(*cp["rating_range"]), 2)
                review_count = self.rng.randint(*cp["review_count_range"])
                popularity = max(
                    cp["popularity_floor"],
                    review_count * self.rng.uniform(*cp["popularity_multiplier_range"]),
                )
                catalog.append(
                    Place(
                        place_id=f"pl_{place_index}",
                        name=f"{category}_{place_index}",
                        category=category,
                        location=location,
                        keywords=tuple(keywords),
                        rating=rating,
                        review_count=review_count,
                        popularity=popularity,
                    )
                )
        return catalog

    def _build_place_catalog(self):
        loaded = self._load_osm_poi_catalog(self.poi_csv_path)
        if loaded:
            return loaded
        return self._build_synthetic_osm_catalog()

    # ── Mode choice infrastructure ───────────────────────────────────────────

    def _peer_status_average(self):
        total = sum(self.last_mode_counts.values())
        if total == 0:
            return 0.5
        return sum(self.mode_status[m] * c for m, c in self.last_mode_counts.items()) / total

    def _plan_agents_for_day(self, day_index):
        rs_for_agent = self.recommender_stack if self.use_recommenders else None
        for agent in self.agents:
            agent.plan_day(self, self.rng, recommender_stack=rs_for_agent, day_index=day_index)

    def run_days(self, num_days=7, progress=None, on_day_complete=None, should_stop=None):
        """Run ``num_days`` days. If ``should_stop`` is given and returns True at
        the start of a day, raise :class:`SimulationCancelled` (used for clean
        mid-run cancellation; has no effect on results when unused)."""
        outputs = []
        for day in range(num_days):
            if should_stop is not None and should_stop():
                raise SimulationCancelled()
            if progress is not None:
                progress(day + 1, num_days)
            if day > 0:
                self._plan_agents_for_day(day_index=day)
            self.last_road_volume = 0
            self.last_transit_volume = 0
            self.last_mode_counts = Counter()
            self.run_day()
            day_summary = self.summarize().copy()
            day_summary["day"] = day + 1
            outputs.append(day_summary)
            # Fires while ``agent.trips`` still holds this day's trips (the next
            # day's planning resets them) — lets the caller capture per-day viz.
            if on_day_complete is not None:
                on_day_complete(day, self)
        return outputs

    def _mode_available(self, mode, agent, distance_km):
        ma = params.MODE_AVAILABILITY
        # TOGGLE: CAR_ONLY_MODE — only car is ever available.
        if params.SIMPLIFICATION_TOGGLES.get("CAR_ONLY_MODE", False):
            return mode == "car"
        if mode == "car":
            if agent.car_access_type == "own car":
                return True
            if agent.car_access_type == "carshare":
                return distance_km >= ma["carshare_min_distance_km"]
            return distance_km >= ma["no_car_min_distance_km"]
        if mode == "bike":
            # No ownership gate: bike-share (e.g., Citi Bike) makes cycling
            # available to any able rider within range. ACS has no bike-
            # ownership variable, so we don't model one.
            if agent.characteristics["age"] > ma["bike_max_age"]:
                return False
            if agent.mobility_needs == "ADA/wheelchair":
                return False
            return distance_km <= ma["bike_max_distance_km"]
        if mode == "walk":
            if agent.characteristics["age"] > ma["walk_max_age"]:
                return False
            walk_limit = max(ma["walk_min_distance_km"], agent.walk_tolerance_min / ma["walk_speed_for_tolerance"])
            if agent.mobility_needs == "ADA/wheelchair":
                walk_limit = min(walk_limit, ma["walk_ada_max_km"])
            return distance_km <= walk_limit
        return True

    def _road_congestion_factor(self, volume):
        cp = params.CONGESTION_PARAMS
        x = max(0.0, volume / self.road_capacity)
        return 1.0 + cp["bpr_alpha"] * (x ** cp["bpr_beta"])

    def _transit_crowding_factor(self, volume):
        cp = params.CONGESTION_PARAMS
        x = max(0.0, volume / self.transit_capacity)
        return 1.0 + cp["transit_crowding_coeff"] * (x ** cp["transit_crowding_exponent"])

    def _weather_speed_factor(self, mode):
        weather = self.context["weather"]
        return params.WEATHER_FACTORS.get(weather, {}).get(mode, 1.0)

    def _policy_cost_adjustment(self, mode, distance_km):
        policy = self.context["ai_intervention"]
        cp = params.CONGESTION_PARAMS
        if policy == "congestion_pricing" and mode == "car":
            return cp["congestion_pricing_base"] + cp["congestion_pricing_per_km"] * distance_km
        return 0.0

    def _base_mode_utility(self, agent, mode, distance_km, road_factor, transit_factor):
        uw = params.UTILITY_WEIGHTS
        spec = self.modes[mode]
        speed = spec["speed_kmh"] * self._weather_speed_factor(mode)
        in_vehicle = (distance_km / max(1e-3, speed)) * 60
        if mode == "car":
            in_vehicle *= road_factor
        if mode == "transit":
            in_vehicle *= transit_factor
        wait = spec["wait_min"]
        travel_time = in_vehicle + wait

        monetary_cost = spec["fixed_cost"] + spec["cost_per_km"] * distance_km
        monetary_cost += self._policy_cost_adjustment(mode, distance_km)

        gen_cost = (travel_time / 60) * agent.vot * agent.preferences["time"]
        gen_cost += monetary_cost * agent.preferences["cost"]

        emissions = spec["emissions_g_per_km"] * distance_km
        comfort = spec["comfort"]
        green_weight = agent.preferences["green"]
        if self.context["social_norms"] == "green":
            green_weight *= uw["green_norm_boost"]

        utility = -gen_cost / uw["gen_cost_denominator"] + comfort * agent.preferences["comfort"]
        utility -= green_weight * (emissions / uw["emissions_divisor"])
        return utility, travel_time, monetary_cost, emissions, gen_cost

    def _leisure_mode_adjustment(self, mode, next_activity):
        if next_activity.type != "leisure" or not next_activity.subtype:
            return 0.0
        return params.LEISURE_MODE_TASTE_SHIFTS.get(next_activity.subtype, {}).get(mode, 0.0)

    def _mode_outcome(self, agent, mode, distance_km, road_factor, transit_factor, peer_status, next_activity):
        """Utility/cost outcome for a single mode (shared by evaluation + fallback)."""
        uw = params.UTILITY_WEIGHTS
        utility, travel_time, monetary_cost, emissions, gen_cost = self._base_mode_utility(
            agent, mode, distance_km, road_factor, transit_factor
        )

        if next_activity.is_mandatory:
            derived = agent.motivation_weights["derived"]
            utility -= uw["derived_penalty_coeff"] * derived * (travel_time / uw["derived_time_divisor"])

        intrinsic = agent.motivation_weights["intrinsic"]
        utility += intrinsic * self.mode_enjoyment[mode] * (travel_time / uw["intrinsic_time_divisor"])

        escape = agent.motivation_weights["escape"]
        if next_activity.type == "leisure":
            utility += escape * min(uw["escape_cap"], distance_km / uw["escape_distance_scale_km"])
            utility += self._leisure_mode_adjustment(mode, next_activity)

        positionality = agent.motivation_weights["positionality"]
        status_delta = self.mode_status[mode] - peer_status
        utility += positionality * agent.attitudes["status_seeking"] * status_delta

        practice_bias = self.community_mode_bias(mode)
        utility += practice_bias * agent.attitudes["practice_conformity"]
        utility += agent.mode_preference_bias.get(mode, 0.0)

        if mode == "car":
            utility -= agent.car_access_penalty

        travel_utility = utility
        activity_utility = 0.0
        if next_activity.type == "leisure" and next_activity.subtype:
            seg_cfg = params.LEISURE_SEGMENTS.get(next_activity.subtype, {})
            activity_utility += seg_cfg.get("activity_utility", 0.0)
            activity_utility += uw["activity_intrinsic_coeff"] * agent.motivation_weights["intrinsic"]
            activity_utility += uw["activity_escape_coeff"] * agent.motivation_weights["escape"]
        utility = travel_utility + activity_utility

        return {
            "utility": utility,
            "travel_utility": travel_utility,
            "activity_utility": activity_utility,
            "travel_time": travel_time,
            "cost": monetary_cost,
            "emissions": emissions,
            "gen_cost": gen_cost,
            "distance_km": distance_km,
        }

    def _evaluate_modes(self, agent, origin, destination, current_activity, next_activity):
        distance_km = self.distance_km(origin, destination)
        road_factor = self._road_congestion_factor(self.last_road_volume)
        transit_factor = self._transit_crowding_factor(self.last_transit_volume)
        peer_status = self._peer_status_average()

        outcomes = {}
        for mode in self.modes:
            if not self._mode_available(mode, agent, distance_km):
                continue
            outcomes[mode] = self._mode_outcome(
                agent, mode, distance_km, road_factor, transit_factor, peer_status, next_activity
            )

        if not outcomes:
            # No mode passed availability (e.g., a carless agent on a mid-range
            # trip once transit is disabled). Walking is always physically
            # possible, so use it as the universal fallback.
            fallback = "walk" if "walk" in self.modes else next(iter(self.modes))
            outcomes[fallback] = self._mode_outcome(
                agent, fallback, distance_km, road_factor, transit_factor, peer_status, next_activity
            )

        return outcomes

    # ── Decision paradigms ───────────────────────────────────────────────────

    def _choose_mode_utility(self, outcomes):
        modes = list(outcomes.keys())
        utilities = [outcomes[m]["utility"] for m in modes]
        probs = softmax(utilities)
        return self.rng.choices(modes, weights=probs, k=1)[0]

    def _choose_mode_regret(self, outcomes):
        regrets = {}
        for m_i, d_i in outcomes.items():
            regret = 0.0
            for m_j, d_j in outcomes.items():
                if m_i == m_j:
                    continue
                regret += max(0.0, d_j["gen_cost"] - d_i["gen_cost"])
            regrets[m_i] = regret
        return min(regrets, key=regrets.get)

    def _choose_mode_prospect(self, outcomes, agent):
        ref_time = agent.reference_points["time_min"]
        ref_cost = agent.reference_points["cost"]

        best_mode = None
        best_score = -1e9
        for mode, data in outcomes.items():
            loss_time = max(0.0, data["travel_time"] - ref_time)
            gain_time = max(0.0, ref_time - data["travel_time"])
            loss_cost = max(0.0, data["cost"] - ref_cost)
            gain_cost = max(0.0, ref_cost - data["cost"])

            loss_penalty = agent.loss_aversion * (loss_time / 60) * agent.vot
            loss_penalty += agent.loss_aversion * loss_cost
            gain_reward = (gain_time / 60) * agent.vot + gain_cost

            score = data["utility"] + (gain_reward - loss_penalty) / 8.0
            if score > best_score:
                best_score = score
                best_mode = mode

        return best_mode

    def _choose_mode_satisficing(self, outcomes, agent):
        for mode, data in outcomes.items():
            if data["gen_cost"] <= agent.satisficing_threshold:
                return mode
        return self._choose_mode_utility(outcomes)

    def choose_mode(self, agent, current_activity, next_activity):
        # TOGGLE: CAR_ONLY_MODE disables the mode-choice model entirely.
        # Every trip is assigned to the car without any evaluation of
        # alternative modes, and without any dependence on the agent's
        # decision paradigm.
        if params.SIMPLIFICATION_TOGGLES.get("CAR_ONLY_MODE", False):
            return "car"
        outcomes = self._evaluate_modes(agent, current_activity.location, next_activity.location, current_activity, next_activity)
        paradigm = agent.decision_paradigms.get("mode_choice", "utility")
        if paradigm == "regret":
            return self._choose_mode_regret(outcomes)
        if paradigm == "prospect":
            return self._choose_mode_prospect(outcomes, agent)
        if paradigm == "satisficing":
            return self._choose_mode_satisficing(outcomes, agent)
        if paradigm == "habit" and agent.habit_mode in outcomes:
            return agent.habit_mode
        return self._choose_mode_utility(outcomes)

    def _record_recommendation_feedback(self, agent, feedback_trip, feedback_params):
        liked, p_like = agent.evaluate_recommendation_feedback(feedback_trip, self, self.rng)
        feedback_trip.user_feedback_like = liked
        # Public review: the like/dislike also moves the POI's live rating and
        # review count, which every recommender scores with from now on.
        self.place_dynamics.record_feedback(feedback_trip.place_id, bool(liked))
        feedback_strength = clamp(
            max(
                feedback_params["feedback_strength_floor"],
                abs(2.0 * p_like - 1.0) * agent.feedback_sensitivity,
            ),
            feedback_params["feedback_strength_min"],
            feedback_params["feedback_strength_max"],
        )
        self.recommender_stack.record_feedback(
            user_id=agent.id,
            source=feedback_trip.recommendation_source,
            place_id=feedback_trip.recommended_place_id,
            liked=bool(liked),
            feedback_strength=feedback_strength,
        )
        if hasattr(self.recommender_stack, "record_continuous_feedback"):
            from .welfare_layer import compute_continuous_feedback

            continuous_feedback = compute_continuous_feedback(
                feedback_trip,
                p_like=p_like,
                feedback_sensitivity=getattr(agent, "feedback_sensitivity", 1.0),
            )
            self.recommender_stack.record_continuous_feedback(
                continuous_feedback,
                travel_time_min=feedback_trip.travel_time_min,
                monetary_cost=feedback_trip.cost,
            )

    # ── Day simulation ───────────────────────────────────────────────────────

    def run_day(self, record_history=False):
        sd = params.SIM_DEFAULTS
        fp = params.FEEDBACK_PARAMS
        stats = {
            "trips": 0,
            "mode_counts": Counter(),
            "total_travel_time": 0.0,
            "total_cost": 0.0,
            "total_emissions": 0.0,
            "total_delay": 0.0,
            "late_arrivals": 0,
        }
        history = [] if record_history else None

        time_bins = range(0, 1440, self.time_step)
        for t in time_bins:
            # Arrivals
            for agent in self.agents:
                if agent.in_transit and agent.arrival_time is not None and t >= agent.arrival_time:
                    agent.in_transit = False
                    agent.arrival_time = None
                    agent.current_trip = None
                    agent.current_activity_index += 1
                    if agent.current_activity_index >= len(agent.schedule):
                        continue
                    activity = agent.schedule[agent.current_activity_index]
                    actual_start = max(t, activity.start_time)
                    delay = max(0, t - activity.start_time)
                    agent.total_delay += delay
                    if delay > 0:
                        stats["late_arrivals"] += 1
                    agent.activity_end_time = actual_start + activity.duration
                    # Footfall: arriving at a catalog POI counts as a visit
                    # (recommended or organic), feeding its live popularity.
                    if activity.type == "leisure" and activity.place_id:
                        self.place_dynamics.record_visit(activity.place_id)

            # Departures
            departures = []
            for agent in self.agents:
                if agent.in_transit:
                    continue
                if agent.current_activity_index >= len(agent.schedule) - 1:
                    continue
                if t >= agent.activity_end_time:
                    current_activity = agent.schedule[agent.current_activity_index]
                    if (
                        self.use_recommenders
                        and current_activity.type == "leisure"
                        and current_activity.accepted_recommendation
                        and current_activity.recommended_place_id
                    ):
                        feedback_trip = None
                        for tr in reversed(agent.trips):
                            if (
                                tr.purpose == "leisure"
                                and tr.recommended_place_id == current_activity.recommended_place_id
                                and tr.user_feedback_like == -1
                            ):
                                feedback_trip = tr
                                break
                        if feedback_trip is not None:
                            self._record_recommendation_feedback(agent, feedback_trip, fp)
                    next_activity = agent.schedule[agent.current_activity_index + 1]
                    if current_activity.location != next_activity.location:
                        departures.append((agent, current_activity, next_activity))
                    else:
                        agent.current_activity_index += 1
                        agent.activity_end_time = t + next_activity.duration
                        # Zero-distance transition into a POI still counts as a visit.
                        if next_activity.type == "leisure" and next_activity.place_id:
                            self.place_dynamics.record_visit(next_activity.place_id)

            chosen = []
            road_volume = 0
            transit_volume = 0
            step_mode_counts = Counter()
            step_totals = {"travel_time": 0.0, "cost": 0.0, "emissions": 0.0}

            if departures:
                for agent, current_activity, next_activity in departures:
                    mode = self.choose_mode(agent, current_activity, next_activity)
                    chosen.append((agent, current_activity, next_activity, mode))

                road_volume = sum(1 for *_agent, _cur, _nxt, mode in chosen if mode == "car")
                transit_volume = sum(1 for *_agent, _cur, _nxt, mode in chosen if mode == "transit")

                alpha = sd["reference_point_smoothing"]
                for agent, current_activity, next_activity, mode in chosen:
                    outcomes = self._evaluate_modes(agent, current_activity.location, next_activity.location, current_activity, next_activity)
                    data = outcomes[mode]
                    travel_time_min = max(1, int(math.ceil(data["travel_time"])))
                    arrival_time = t + travel_time_min
                    agent.in_transit = True
                    agent.arrival_time = arrival_time
                    agent.current_trip = {
                        "origin": current_activity.location,
                        "destination": next_activity.location,
                        "depart_time": t,
                        "arrival_time": arrival_time,
                    }

                    agent.reference_points["time_min"] = (1.0 - alpha) * agent.reference_points["time_min"] + alpha * travel_time_min
                    agent.reference_points["cost"] = (1.0 - alpha) * agent.reference_points["cost"] + alpha * data["cost"]
                    agent.habit_mode = mode

                    trip = Trip(
                        agent_id=agent.id,
                        origin=current_activity.location,
                        destination=next_activity.location,
                        depart_time=t,
                        mode=mode,
                        purpose=next_activity.type,
                        purpose_subtype=next_activity.subtype,
                        recommendation_source=next_activity.planned_source if next_activity.type == "leisure" else "organic",
                        accepted_recommendation=next_activity.accepted_recommendation if next_activity.type == "leisure" else False,
                        eta_acceptance=agent.last_eta if next_activity.type == "leisure" else 0.0,
                        recommended_place_id=next_activity.recommended_place_id if next_activity.type == "leisure" else "",
                        place_id=next_activity.place_id,
                        user_feedback_like=-1,
                        distance_km=data["distance_km"],
                        travel_time_min=travel_time_min,
                        cost=data["cost"],
                        emissions_g=data["emissions"],
                        utility=data["utility"],
                        travel_utility=data["travel_utility"],
                        activity_utility=data["activity_utility"],
                        arrival_time=arrival_time,
                        gen_cost=data["gen_cost"],
                    )
                    agent.trips.append(trip)

                    stats["trips"] += 1
                    stats["mode_counts"][mode] += 1
                    stats["total_travel_time"] += travel_time_min
                    stats["total_cost"] += data["cost"]
                    stats["total_emissions"] += data["emissions"]

                    step_mode_counts[mode] += 1
                    step_totals["travel_time"] += travel_time_min
                    step_totals["cost"] += data["cost"]
                    step_totals["emissions"] += data["emissions"]

                self.last_road_volume = road_volume
                self.last_transit_volume = transit_volume
                self.last_mode_counts = Counter(mode for *_rest, mode in chosen)

            if record_history:
                step = {
                    "time": t,
                    "departures": len(departures),
                    "trips_started": len(chosen),
                    "road_volume": road_volume,
                    "transit_volume": transit_volume,
                    "mode_counts": step_mode_counts,
                    "avg_travel_time_min": (step_totals["travel_time"] / len(chosen)) if chosen else 0.0,
                    "avg_cost": (step_totals["cost"] / len(chosen)) if chosen else 0.0,
                    "avg_emissions_g": (step_totals["emissions"] / len(chosen)) if chosen else 0.0,
                    "road_congestion_factor": self._road_congestion_factor(road_volume),
                    "transit_crowding_factor": self._transit_crowding_factor(transit_volume),
                    "in_transit": sum(1 for a in self.agents if a.in_transit),
                }
                history.append(step)

        stats["total_delay"] = sum(a.total_delay for a in self.agents)
        self.stats = stats
        if record_history:
            return stats, history
        return stats

    # ── Summary ──────────────────────────────────────────────────────────────

    def summarize(self):
        if not self.stats:
            return {}
        stats = self.stats
        trips = max(1, stats["trips"])
        purpose_counts = Counter()
        leisure_subtype_counts = Counter()
        rec_source_counts = Counter()
        rec_accepted = 0
        eta_values = []
        feedback_likes = 0
        feedback_count = 0
        total_trip_utility = 0.0
        total_travel_utility = 0.0
        total_activity_utility = 0.0
        for agent in self.agents:
            for trip in agent.trips:
                purpose_counts[trip.purpose] += 1
                total_trip_utility += trip.utility
                total_travel_utility += trip.travel_utility
                total_activity_utility += trip.activity_utility
                if trip.purpose == "leisure" and trip.purpose_subtype:
                    leisure_subtype_counts[trip.purpose_subtype] += 1
                    rec_source_counts[trip.recommendation_source] += 1
                    if trip.accepted_recommendation:
                        rec_accepted += 1
                    eta_values.append(trip.eta_acceptance)
                    if trip.user_feedback_like in (0, 1):
                        feedback_count += 1
                        feedback_likes += trip.user_feedback_like
        leisure_trips = max(1, sum(leisure_subtype_counts.values()))
        summary = {
            "trips": stats["trips"],
            "avg_travel_time_min": stats["total_travel_time"] / trips,
            "avg_cost": stats["total_cost"] / trips,
            "avg_emissions_g": stats["total_emissions"] / trips,
            "avg_delay_min": stats["total_delay"] / max(1, len(self.agents)),
            "late_arrivals": stats["late_arrivals"],
            "mode_share": {k: v / trips for k, v in stats["mode_counts"].items()},
            "purpose_share": {k: v / trips for k, v in purpose_counts.items()},
            "leisure_subtype_counts": dict(leisure_subtype_counts),
            "recommendation_source_counts": dict(rec_source_counts),
            "recommendation_acceptance_rate": rec_accepted / leisure_trips,
            "avg_eta": (sum(eta_values) / len(eta_values)) if eta_values else 0.0,
            "feedback_like_rate": (feedback_likes / feedback_count) if feedback_count else 0.0,
            "total_trip_utility": total_trip_utility,
            "avg_trip_utility": total_trip_utility / trips,
            "total_travel_utility": total_travel_utility,
            "total_activity_utility": total_activity_utility,
        }
        return summary

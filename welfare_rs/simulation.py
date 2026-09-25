"""Simulation orchestrator for the travel-behaviour ABM."""

from __future__ import annotations

import copy
import csv
import hashlib
import itertools
import math
import random
import warnings
from collections import Counter
from pathlib import Path

from typing import NamedTuple

from . import params, tastes
from . import agent as agent_module
from .agent import Agent
from .datastructures import Trip
from .poi_select import select_catalog_rows
from .utils import clamp, haversine_km, softmax
from .recommender_systems import (
    ConfigurableRecommender,
    Place,
    PlaceDynamics,
    RecommenderConfig,
    SingleRecommenderOrchestrator,
    build_recommender_stack,
)


class _Stop(NamedTuple):
    """The activity a planned leg ends at — the fields mode utility reads."""

    type: str
    subtype: str
    is_mandatory: bool
    place_id: str = ""


# Car owners take the car with them (chain-bound); everyone else's "car" is a
# ride that does not have to come home.
_CHAIN_BOUND_CAR = ("own car", "carshare")


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
        persona_rows=None,
        population_rows=None,
        poi_csv_path=None,
        poi_rows=None,
        road_network=None,
        disabled_modes=None,
        recommender_override=None,
        recommender_factory=None,
        recommender_config=None,
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
        # The builder-defined recommender for this run. None keeps the legacy
        # four-platform stack (build_recommender_stack), which is retained so
        # existing notebooks and scripts still work.
        self.recommender_config = recommender_config
        self.eta_shift = eta_shift
        self.use_persona_agents = use_persona_agents
        self.persona_csv_path = persona_csv_path if persona_csv_path is not None else params.NYC_PERSONA_CSV_PATH
        # Persona rows may be handed in directly, the same way poi_rows are, so
        # a datasource-driven backend can serve a population that is not a file
        # (PostgresDataSource.personas() reads the survey.personas view). Takes
        # precedence over persona_csv_path.
        self.persona_rows = persona_rows
        # A pre-built ACS/PUMS population (db/build_population.py). When given,
        # it decides the agents entirely: geography, demographics and which
        # survey respondent each agent takes its psychometrics from. Nothing is
        # sampled here — see DataSource.population for why that work is offline.
        self.population_rows = list(population_rows) if population_rows else None
        self.poi_csv_path = poi_csv_path
        # Real-POI rows may be handed in directly (datasource-driven backends);
        # they take precedence over reading poi_csv_path.
        self.poi_rows = poi_rows
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
        # drops modes entirely; by default that is the transit stand-in, which
        # is a flat speed over road distance rather than a transit network and
        # is not offered until one exists.
        self.modes = {k: dict(v) for k, v in params.MODE_PARAMS.items()}
        for _m in (params.DISABLED_MODES if disabled_modes is None else disabled_modes):
            self.modes.pop(_m, None)
        self.mode_status = dict(params.MODE_STATUS)
        self.mode_enjoyment = dict(params.MODE_ENJOYMENT)

        # Walking and cycling run on their own core networks, attached to the
        # drive graph by metro.build_metro_network. A mode without its network
        # is not offered at all — pricing it on the drive graph would route
        # pedestrians along one-way arterials and never through a park.
        self.metro = getattr(road_network, "metro", None)
        attached = getattr(road_network, "mode_networks", None) or {}
        self.active_networks = {m: attached[m] for m in params.ACTIVE_NETWORK_PARAMS["modes"]
                                if m in self.modes and m in attached}
        missing = [m for m in params.ACTIVE_NETWORK_PARAMS["modes"]
                   if m in self.modes and m not in attached]
        for m in missing:
            self.modes.pop(m)
        if missing:
            warnings.warn(
                f"{road_network.name}: no {'/'.join(missing)} network attached, so "
                f"{' and '.join(missing)} {'is' if len(missing) == 1 else 'are'} not "
                "offered in this run. Build them with scripts/warm_metros.py.",
                RuntimeWarning, stacklevel=2,
            )
        self._mode_asc = dict(params.MODE_ASC.get(self.metro, {}))
        # Legs for which no permitted mode existed at departure. The population
        # filter and the planner are built so this stays zero; it is counted,
        # not silently absorbed, so a run can prove it.
        self.infeasible_legs = 0
        self.dropped_population_rows = 0
        self.dropped_outside_graph = 0

        # Place catalog and agents must exist before some recommender factories
        # can finish wiring treatment-specific stacks (for example Oracle).
        self.place_catalog = self._build_place_catalog()
        self.place_by_id = {p.place_id: p for p in self.place_catalog}
        # Empirical taste mix of the catalog, which agent tastes are drawn from.
        # Derived from the POIs themselves, so it reflects this metro's actual
        # supply rather than an assumed one.
        self.taste_distribution = tastes.taste_distribution(
            [(p.place_id, p.name, p.category) for p in self.place_catalog],
            {p.place_id: p.taste_tags for p in self.place_catalog},
        )

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

        # Read by mode utility, which the planner now calls (the logsum), so it
        # has to exist before agents plan day 0.
        self.last_road_volume = 0
        self.last_transit_volume = 0
        self.last_mode_counts = Counter()
        # Peer mode status an agent expects when planning: yesterday's mix, or
        # neutral on the first day.
        self._planning_peer_status = 0.5

        _num_agents = num_agents if num_agents is not None else sd["num_agents"]
        self.agents = []
        personas = self._load_personas() if self.use_persona_agents else []

        # With a built population, personas are looked up BY RESPONDENT ID
        # rather than by position: the population already recorded which
        # respondent each agent matched, on sex, employment, income and age.
        self._persona_by_id = {}
        if self.population_rows:
            self._persona_by_id = {str(p.get("PersonaID")): p for p in personas}
            missing = {r["respondent_id"] for r in self.population_rows
                       if str(r["respondent_id"]) not in self._persona_by_id}
            if missing:
                raise ValueError(
                    f"population references {len(missing)} respondent id(s) absent "
                    f"from the survey population (first: {sorted(missing)[:3]}). "
                    "The population and the survey view are out of step — rebuild "
                    "with db/build_population.py.")
            if len(self.population_rows) < _num_agents:
                raise ValueError(
                    f"population holds {len(self.population_rows):,} agents but the "
                    f"run asks for {_num_agents:,}. Build more with "
                    "db/build_population.py --agents.")
            # Two kinds of row are left out, in row order, and the next rows
            # take their places:
            # * homes in a county the graph does not cover. The drive graph and
            #   its commute pairs are restricted to the counties holding 95% of
            #   the metro's commuters (the rest is "real LODES data with
            #   unusable geometry"), but the population build drew homes from
            #   every county, so these agents snapped to the graph's edge and
            #   were charged a 7-18 km (up to 68 km) straight-line access leg.
            # * people with no permitted way to reach their workplace.
            # Every condition of a study applies the same rules to the same
            # rows, so the paired No-RS comparison still pairs identical agents.
            graph_counties = self._graph_counties()
            eligible = []
            for row in self.population_rows:
                if len(eligible) == _num_agents:
                    break
                if graph_counties is not None and str(row["home_geoid"])[:5] not in graph_counties:
                    self.dropped_outside_graph += 1
                    self.dropped_population_rows += 1
                elif self._population_row_eligible(row):
                    eligible.append(row)
                else:
                    self.dropped_population_rows += 1
            if len(eligible) < _num_agents:
                raise ValueError(
                    f"only {len(eligible):,} of {len(self.population_rows):,} population "
                    f"rows live on the graph and can travel under the mode rules, but "
                    f"the run asks for {_num_agents:,}. Pass more rows.")
            self.population_rows = eligible
        # The given personas are used as-is for the first len(personas) agents.
        # Beyond that, whole profiles are resampled with replacement (see
        # _resample_persona) so the population can exceed the provided set
        # without breaking the joint distribution over its attributes.
        used_homes = set()
        for p in personas:
            try:
                used_homes.add((round(float(p["start_latitude"]), 6), round(float(p["start_longitude"]), 6)))
            except (KeyError, ValueError, TypeError):
                pass
        syn_rng = random.Random(self.seed + 991)
        tp = params.TASTE_PARAMS
        for i in range(_num_agents):
            if self.population_rows:
                agent = self._build_agent_from_population(i, self.population_rows[i])
            elif personas:
                persona = personas[i] if i < len(personas) else self._resample_persona(i, personas, syn_rng, used_homes)
                agent = self._build_agent_from_persona(i, persona)
            else:
                agent = self._build_random_agent(i)
            # Latent taste, on its own RNG stream keyed by (seed, agent id).
            # Agent i must hold the SAME taste under every condition or the
            # paired No-RS comparison is comparing different people; a dedicated
            # stream also keeps the draw from shifting when unrelated model
            # changes consume a different number of values from self.rng.
            #
            # Seeded from a STRING, not an arithmetic combination. Mersenne
            # Twister states from consecutive integer seeds are correlated in
            # their first draws — Random(n) and Random(n+1) opened 0.2636 and
            # 0.2824 here — so ``seed * k + i`` gave neighbouring agents
            # near-identical tastes. Agents are built in persona order, so that
            # correlation would have run straight through every by-segment
            # metric. Random() hashes a str seed (SHA-512) and decorrelates.
            agent.tastes = tastes.sample_agent_tastes(
                self.taste_distribution,
                random.Random(f"tastes:{self.seed}:{i}"),
                beta=float(tp["beta"]),
                per_family=int(tp["per_family"]),
            )
            self.agents.append(agent)

        # Routing only ever runs between agent homes/works and catalog POIs;
        # registering them lets the network prune each cached Dijkstra tree to
        # just these target nodes (rare fallback destinations self-heal lazily).
        self.road_network.register_route_targets(
            [p.location for p in self.place_catalog]
            + [a.home for a in self.agents]
            + [a.work for a in self.agents]
        )
        core_agents = [a for a in self.agents if a.lives_in_core]
        for _net in self.active_networks.values():
            _net.register_route_targets(
                [p.location for p in self.place_catalog]
                + [a.home_true for a in core_agents]
                + [a.work_true for a in core_agents]
            )

        self.recommender_stack = self._resolve_recommender_stack(
            recommender_override=recommender_override,
            recommender_factory=recommender_factory,
        )
        self._plan_agents_for_day(day_index=0)

        self.stats = None

    # ── Environment interface (shared with Agent.plan_day) ───────────────────

    def distance_km(self, a, b):
        """Network shortest-path distance in km between two (lat, lon) nodes."""
        return self.road_network.route_length_km_latlon(a, b)

    def community_mode_bias(self, mode):
        """Return a social-practice bias for a mode (SPT)."""
        return self.context.get("community_mode_bias", {}).get(mode, 0.0)

    def taste_bonus(self, agent, place_id):
        """Extra activity utility when ``place_id`` matches the agent's taste.

        This is the ground truth the recommenders do not observe. It is what
        makes "which POI" a question a recommender can be right or wrong about:
        without it every candidate of a subtype yields the same activity
        utility to a given agent, and the only thing a recommendation can
        change is how far the agent travels.
        """
        if not place_id:
            return 0.0
        place = self.place_by_id.get(place_id)
        if place is None or not place.taste_tags:
            return 0.0
        agent_tastes = getattr(agent, "tastes", None)
        if not agent_tastes:
            return 0.0
        if tastes.taste_match(agent_tastes, place.taste_tags):
            return float(params.TASTE_PARAMS["match_bonus"])
        return 0.0

    def sample_poi(self, subtype, origin, rng, max_km=None):
        """Sample an organic destination POI for a leisure subtype.

        Among the catalog POIs of the chosen subtype, a place is drawn with
        probability ``∝ exp(-dist/scale)`` so closer options dominate — the
        proximity-driven "self-selected alternative" from the paper. Distances
        use straight-line (haversine) proximity, mirroring the recommenders'
        proximity heuristic; realized travel cost is still measured on the
        network downstream.

        ``max_km`` restricts the draw to places within that straight-line
        distance of ``origin`` — the agent's physical reach when it has no car
        (see :meth:`reach_limit_km`). A carless agent picks among the places it
        could get to, not among all of them.

        Returns ``(location, place_id)``; falls back to a random intersection
        (with ``place_id=""``) when the catalog has no POI for the subtype, and
        returns ``(None, "")`` when none of them is within ``max_km``.
        """
        places = self._pois_by_subtype.get(subtype) or []
        if not places:
            # Fallback destinations stay in the principal city on metro networks.
            return self.road_network.sample_core_latlon(self._loc_rng), ""
        if max_km is not None:
            places = [p for p in places if haversine_km(origin, p.location) <= max_km]
            if not places:
                return None, ""
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

    def _load_personas(self):
        """The agent population, from handed-in rows or the persona CSV.

        Rows passed to the constructor win: on the lab server they come from
        ``PostgresDataSource.personas()`` (the ``survey.personas`` view), which
        yields the same dicts the CSV does — string values, empty string for a
        missing cell — so nothing below this point can tell the two apart.
        Order is preserved either way, and it matters: agent ``i`` is
        ``personas[i]``, and the paired No-RS comparison needs that to be the
        same respondent in every run.
        """
        if self.persona_rows is not None:
            return self._warn_if_synthetic([dict(r) for r in self.persona_rows
                                            if r.get("PersonaID")])
        if not self.persona_csv_path:
            return []
        path = Path(self.persona_csv_path)
        if not path.exists():
            return []
        rows = []
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("PersonaID"):
                    rows.append(row)
        return self._warn_if_synthetic(rows)

    def _home_from_latlon(self, lat, lon):
        """Map a persona's real (lat, lon) to the nearest network node."""
        return self.road_network.snap_latlon(lat, lon)

    def _sample_home_work(self, persona=None, age_years=None, annual_income=None):
        """Return ``(home, work, home_access_km, work_access_km, home_true,
        work_true)`` for one agent — the ``_true`` points being the unsnapped
        locations (equal to the snapped ones when the sampler draws nodes).

        Preference order:

        1. A **real LODES commute pair** when the network carries them — one
           draw ∝ that pair's job count gives a home and a workplace that
           actually go together, so commute lengths follow the published
           distribution instead of two independent uniform draws. ``age_years``
           and ``annual_income`` narrow that draw to the matching LODES worker
           segments, keeping commute geography consistent with the agent's
           demographics. Pass ``annual_income`` only for agents who hold a job:
           LODES earnings are job earnings, so a non-earner has nothing to
           segment on.
        2. Otherwise the provisional split: a county-weighted home (or the
           persona's own coordinates on an unlayered network) plus a uniform
           workplace inside the core.
        """
        net = self.road_network
        drawn = (net.sample_commute(self._loc_rng, age_years=age_years,
                                    annual_income=annual_income, with_true=True)
                 if net.has_commutes else None)
        if drawn is not None:
            return drawn

        if net.has_layers or persona is None:
            # Two-layer metro: personas supply behaviour only (their stored
            # coordinates belong to another city).
            home = net.sample_home_latlon(self._loc_rng)
        else:
            try:
                home = self._home_from_latlon(
                    float(persona.get("start_latitude", "")),
                    float(persona.get("start_longitude", "")),
                )
            except ValueError:
                home = net.sample_node_latlon(self._loc_rng)
        # Fallback samplers return graph nodes directly, so there is no snap
        # gap to charge.
        work = net.sample_core_latlon(self._loc_rng)
        return home, work, 0.0, 0.0, home, work

    def _home_work_from_population(self, row):
        """Snap one population row's block coordinates onto the road graph.

        Mirrors :meth:`welfare_rs.geo.RoadNetwork.sample_commute`, including
        what it does with the snap gap. The stored coordinates are census block
        internal points; outside the core the graph carries arterials only, so
        a suburban home can snap 1-3 km. That distance is returned as an
        un-networked access leg and charged by ``_leg_outcomes`` instead of
        letting the trip silently start from the arterial.

        A non-worker has no workplace, and its day is home-anchored
        (:meth:`welfare_rs.agent.Agent.plan_day`). Its work slot is filled with
        its home rather than left empty: every consumer of ``Agent.work``
        expects a point, and a home-anchored day never routes to it.
        """
        from .geo import haversine_km

        net = self.road_network
        home_true = (float(row["home_lat"]), float(row["home_lon"]))
        home = net.snap_latlon(*home_true)
        h_acc = haversine_km(home_true, home)

        if row.get("work_lat") is None or row.get("work_lon") is None:
            return home, home, h_acc, 0.0, home_true, home_true
        work_true = (float(row["work_lat"]), float(row["work_lon"]))
        work = net.snap_latlon(*work_true)
        return home, work, h_acc, haversine_km(work_true, work), home_true, work_true

    @staticmethod
    def _warn_if_synthetic(personas):
        """Announce a fabricated population, every time it is loaded.

        data/make_demo_personas.py exists so the engine can run without the
        IRB-protected survey, and its rows are invented. A run built on them
        must never be mistaken for a run on the survey, so the warning is
        raised rather than logged once: it lands in the output of whatever is
        driving the run.
        """
        n_fake = sum(1 for p in personas if str(p.get("PersonaID", "")).startswith("SYNTHETIC-"))
        if n_fake:
            warnings.warn(
                f"{n_fake} of {len(personas)} personas are SYNTHETIC (fabricated by "
                "data/make_demo_personas.py). This run demonstrates the machinery "
                "and reproduces no published result.",
                RuntimeWarning, stacklevel=2,
            )
        return personas

    @staticmethod
    def _coerce_measurement(value):
        """Parse a persona cell as a number on the item's own response scale.

        Deliberately numeric-only. Categorical tokens ("high", "yes") used to be
        mapped onto [0, 1] here, but the consumer normalises against each item's
        declared scale (``agent.SURVEY_ITEM_SCALES``), so a 0.8 arriving for a
        1-7 item would be read as below its floor. Token-valued persona columns
        are handled in the fallback block, which emits native-scale values.
        """
        if value is None:
            return None
        if isinstance(value, bool):
            return 1.0 if value else 0.0
        if isinstance(value, (int, float)):
            return float(value)
        token = str(value).strip()
        if token == "":
            return None
        try:
            return float(token)
        except ValueError:
            return None

    @staticmethod
    def _unit_to_scale(unit_value, scale_min, scale_max):
        """Place a 0-1 judgement onto an item's native response scale."""
        return scale_min + float(unit_value) * (scale_max - scale_min)

    @staticmethod
    def _persona_is_employed(persona) -> bool:
        """Whether a persona holds a job, from the survey's Employment item (Q5).

        Full-time, part-time and self-employed count; retired, unemployed,
        out-of-the-labour-force and student do not. Part-time is treated as
        employed without proration — the survey records no hours, so any split
        finer than this would be invented. Population files with no Employment
        column fall back to employed, preserving the pre-survey behaviour.
        """
        raw = str(persona.get("Employment", "") or "").strip().lower()
        if raw == "":
            return True
        return raw.startswith("employed") or raw.startswith("self-employed")

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
            "trust_platforms": ("trust_platforms", "lv_trust_platforms", "Recommendation_Trust"),
            # NOT the survey's Autonomy_Control composite. That composite is
            # positively loaded on liking personalisation ("I like when a platform
            # tailors its suggestions", "platforms help me discover things I would
            # not have found") and correlates +0.46 with Recommendation_Trust and
            # +0.43 with following platform recommendations. This slot enters
            # eta_base with a NEGATIVE coefficient (params.SIMPLIFIED_ETA_PARAMS),
            # so feeding it the composite inverts beta_O for the whole population.
            # s6_autonomy_control_a — "I prefer to make my own leisure choices
            # rather than follow platform suggestions" — is the item that measures
            # this construct (-0.52 with trust, -0.29 with follow-through) and is
            # deliberately excluded from every composite in the survey package.
            "autonomy_preference": ("autonomy_preference", "lv_autonomy", "s6_autonomy_control_a"),
            "algorithmic_awareness": (
                "algorithmic_awareness",
                "lv_algorithmic_awareness",
                "Algorithmic_Awareness",
            ),
        }
        # Second alias in each tuple is the source column in the Prolific survey
        # (data_handoff/algo_leisure_survey_2026.csv); see data/build_survey_personas.py.
        item_alias = {
            "local_leisure_frequency": ("local_leisure_frequency", "leisure_freq_num"),
            "overnight_trip_frequency": ("overnight_trip_frequency", "overnight_num"),
            "spontaneity_share": ("spontaneity_share", "spur_num"),
            "follow_through_friend": ("follow_through_friend", "s2_Q16"),
            "follow_through_platform": ("follow_through_platform", "s2_Q17"),
            "follow_through_ai": ("follow_through_ai", "s2_Q18"),
            # Q19's incentive battery has no counterpart in this survey.
            "incentive_coupon": ("incentive_coupon",),
            "incentive_sponsored": ("incentive_sponsored",),
            "incentive_loyalty": ("incentive_loyalty",),
            "cross_platform_search": ("cross_platform_search", "s4_search_c"),
            "price_filter_tendency": ("price_filter_tendency", "s4_search_e"),
            "group_coordination_preference": ("group_coordination_preference", "s4_group_coord_mean"),
            "popularity_herding": ("popularity_herding", "s5_general_a"),
            "unexpected_discovery": ("unexpected_discovery", "s3_attitudes_d"),
            "review_posting": ("review_posting",),
            "social_posting": ("social_posting", "s4_social_media_b"),
            "switch_when_dissatisfied": ("switch_when_dissatisfied",),
            "multi_platform_parallel": ("multi_platform_parallel", "s5_search_c"),
            "advice_goal_directed": ("advice_goal_directed",),
            "objective_algorithmic_literacy": (
                "objective_algorithmic_literacy",
                "s6_algo_knowledge_correct",
            ),
            "city_familiarity": ("city_familiarity", "CityFam_num"),
            "budget_tightness": ("budget_tightness",),
            "ai_itinerary_comfort": ("ai_itinerary_comfort", "s6_trust_reliance_c"),
            "explanation_needed": ("explanation_needed", "s6_trust_reliance_d"),
        }

        for canonical, aliases in big_five_alias.items():
            raw = self._first_nonempty(persona, aliases)
            value = self._coerce_measurement(raw)
            if value is not None:
                profile["big_five"][canonical] = value

        for canonical, aliases in latent_alias.items():
            raw = self._first_nonempty(persona, aliases)
            value = self._coerce_measurement(raw)
            if value is not None:
                profile["latent_variables"][canonical] = value

        for canonical, aliases in item_alias.items():
            raw = self._first_nonempty(persona, aliases)
            value = self._coerce_measurement(raw)
            if value is not None:
                profile["items"][canonical] = value

        # Fall back to the synthetic-persona columns when a survey measurement is
        # absent. These are coarse judgements on [0, 1], so each is placed on its
        # item's native scale before handing over — the consumer normalises
        # against agent.SURVEY_ITEM_SCALES and does not detect pre-normalised
        # values.
        scales = agent_module.SURVEY_ITEM_SCALES
        if "budget_tightness" not in profile["items"]:
            budget = str(persona.get("Budget", "medium")).strip().lower()
            unit = {"low": 0.85, "medium": 0.50, "high": 0.20}.get(budget, 0.50)
            profile["items"]["budget_tightness"] = self._unit_to_scale(unit, *scales["budget_tightness"])
        if "follow_through_ai" not in profile["items"]:
            willingness_ai = str(persona.get("WillingnessAI", "med")).strip().lower()
            unit = {"high": 0.80, "med": 0.55, "low": 0.25}.get(willingness_ai, 0.55)
            profile["items"]["follow_through_ai"] = self._unit_to_scale(unit, *scales["follow_through_ai"])
        if "follow_through_platform" not in profile["items"]:
            profile["items"]["follow_through_platform"] = profile["items"]["follow_through_ai"]
        if "popularity_herding" not in profile["items"]:
            top_rated = str(persona.get("TopRated", "no")).strip().lower() == "yes"
            unit = 0.75 if top_rated else 0.40
            profile["items"]["popularity_herding"] = self._unit_to_scale(unit, *scales["popularity_herding"])
        if "city_familiarity" not in profile["items"]:
            profile["items"]["city_familiarity"] = self._unit_to_scale(0.65, *scales["city_familiarity"])

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
        home, work, h_acc, w_acc, h_true, w_true = self._sample_home_work(
            age_years=age, annual_income=income)
        agent = Agent(i, income, age, car_ownership, home, work, seed=self.seed, eta_shift=self.eta_shift)
        agent.home_access_km, agent.work_access_km = h_acc, w_acc
        self._set_true_anchors(agent, h_true, w_true)
        agent.car_access_type = "own car" if car_ownership else "no car"
        agent.car_access_penalty = 0.0 if car_ownership else ad["no_car_access_penalty"]
        return agent

    def _set_true_anchors(self, agent, home_true, work_true):
        """Record the unsnapped home/work points and core residency."""
        agent.home_true = home_true
        agent.work_true = work_true
        agent.lives_in_core = self.road_network.in_core(*home_true)

    def _build_agent_from_population(self, i, row):
        """Build agent ``i`` from one pre-built population row.

        The row already decided everything the model used to draw at run time:
        which block this agent lives in, which workplace it commutes to (or
        that it has none), how old it is, what it earns, whether its household
        has a car, and which survey respondent it takes its psychometrics from.
        Nothing is sampled here.

        The persona is looked up by the respondent id the build matched, so the
        agent's attitudes belong to someone of its own sex, employment status,
        income band and age band — instead of to whoever happened to sit at
        position ``i`` in the survey file.
        """
        persona = self._persona_by_id[str(row["respondent_id"])]
        return self._build_agent_from_persona(i, persona, population=row)

    def _build_agent_from_persona(self, i, persona, population=None):
        pm = params.PERSONA_MAPPING
        ad = params.AGENT_DEFAULTS

        income_band = str(persona.get("Income", "35-75k")).strip()
        age_band = str(persona.get("Age", "mid adult")).strip()
        inc_lo, inc_hi = pm["income_bands"].get(income_band, (35000, 74000))
        age_lo, age_hi = pm["age_bands"].get(age_band, (30, 55))
        income = self.rng.randint(inc_lo, inc_hi)
        age = self.rng.randint(age_lo, age_hi)

        if population is not None:
            # Measured, not drawn. hh_income is the like-for-like replacement
            # for the survey's household-income band; a group-quarters resident
            # (a dorm or nursing home) has no household income at all, so its
            # own earnings stand in, and only if it has neither does the band
            # draw survive as a last resort.
            for candidate in (population.get("hh_income"), population.get("own_earnings")):
                if candidate is not None:
                    income = int(candidate)
                    break
            age = int(population["age"])

        # Held fixed for the whole population: the survey measures none of these,
        # and their former per-agent draw came from a balanced experimental
        # design rather than any population. See PERSONA_MAPPING.fixed_attributes.
        fa = pm["fixed_attributes"]
        car_access = fa["car_access"]
        transit_access = fa["transit_access"]
        mobility_needs = fa["mobility_needs"]
        transit_access_level = None

        if population is not None:
            # Three of the nine attributes the survey could not ground are
            # measured per agent: household vehicles (PUMS VEH), ambulatory
            # difficulty and the home tract's transit share.
            vehicles = population.get("vehicles")
            if vehicles is not None:
                car_access = "own car" if vehicles > 0 else "none"
            # Ambulatory difficulty specifically (PUMS DPHY), not the general
            # disability flag: hearing, vision and cognitive difficulty do not
            # change how far someone walks to a venue.
            if population.get("ambulatory") is not None:
                mobility_needs = "ADA/wheelchair" if population["ambulatory"] else "none"
            # A neighbourhood measure (ACS B08301 transit share of the home
            # tract), which is what transit access has always meant here.
            if population.get("transit_share") is not None:
                transit_access_level = max(0.0, min(1.0, float(population["transit_share"])))
        risk_salience = fa["risk_salience"]
        walk_tol = int(fa["walk_tolerance_min"])
        primary_interest = fa["primary_interest"]
        time_window = fa["time_window"]

        # Survey-derived: Budget from monthly leisure spend, TopRated from the
        # stated preference for well-known destinations (s5_general_a).
        budget = str(persona.get("Budget", "medium")).strip()
        top_rated = str(persona.get("TopRated", "no")).strip().lower() == "yes"

        # car_access_penalty applies only to the "car" mode, which only car
        # owners (and carshare members) have. A carless agent travels by car
        # as a taxi instead, which pays a metered fare rather than a penalty
        # (Simulation._taxi_leg), so its penalty never fires.
        if car_access == "own car":
            car_ownership = True
            car_access_penalty = 0.0
        elif car_access == "carshare":
            car_ownership = False
            car_access_penalty = pm["carshare_penalty"]
        else:
            car_ownership = False
            car_access_penalty = pm["no_car_penalty"]

        # Non-earners still take a LODES home: worker home locations are the
        # best residential distribution the model has, and the workplace half of
        # the drawn pair simply goes unused by their schedule. Their draw is
        # conditioned on age only — LODES earnings segments are job earnings,
        # which someone without a job does not have.
        if population is not None:
            is_employed = bool(population["employed"])
            home, work, h_acc, w_acc, h_true, w_true = self._home_work_from_population(population)
        else:
            is_employed = self._persona_is_employed(persona)
            home, work, h_acc, w_acc, h_true, w_true = self._sample_home_work(
                persona,
                age_years=age,
                annual_income=income if is_employed else None,
            )

        agent = Agent(i, income, age, car_ownership, home, work, seed=self.seed,
                      eta_shift=self.eta_shift, is_employed=is_employed)
        agent.home_access_km, agent.work_access_km = h_acc, w_acc
        self._set_true_anchors(agent, h_true, w_true)
        if population is not None:
            # Habit starts from how this person really commutes (PUMS JWTRNS).
            # Modes with no counterpart here (transit, worked from home) leave
            # no habit, so the agent chooses by utility until it has one.
            commute_mode = population.get("commute_mode")
            agent.habit_mode = (params.PUMS_JWTRNS_TO_MODE.get(int(commute_mode))
                                if commute_mode is not None else None)

        agent.persona_id = str(persona.get("PersonaID", f"P{i:04d}"))
        agent.car_access_type = car_access
        agent.car_access_penalty = car_access_penalty
        agent.transit_access_level = (
            transit_access_level if transit_access_level is not None
            else pm["transit_access_levels"].get(transit_access, 0.55))
        agent.walk_tolerance_min = max(5, min(45, walk_tol))
        agent.mobility_needs = mobility_needs
        agent.primary_interest = primary_interest
        agent.time_window_pref = time_window
        agent.feedback_sensitivity = 1.1 if risk_salience == "high" else 0.95

        agent.characteristics["car_ownership"] = car_ownership

        # Survey demographics recorded on the agent for reporting and
        # segmentation. These are carried, not behavioural: nothing in the
        # decision rules reads them.
        #
        # Employment is the exception — it is read, above, to decide whether the
        # agent holds a job, and so whether its day is anchored at a workplace
        # (Agent.plan_day) and whether it is charged the household income as its
        # own value of time (Agent.vot). It is kept in this list as well so the
        # raw label survives for reporting.
        for _field in pm["carried_fields"]:
            value = persona.get(_field)
            if value is not None and str(value).strip() != "":
                agent.characteristics[_field] = value

        if population is not None:
            # Provenance: every agent can be traced back to its census
            # geography, its PUMS record and the survey respondent it matched,
            # which is what makes a result auditable rather than merely
            # reproducible. Carried only — no decision rule reads these.
            agent.characteristics.update({
                "population_source": "acs_pums",
                "home_geoid": population["home_geoid"],
                "home_tract": population["home_tract"],
                "puma": population["puma"],
                "pums_serialno": population["serialno"],
                "is_worker": population["is_worker"],
                "own_earnings": population.get("own_earnings"),
                "hh_income": population.get("hh_income"),
                "hh_size": population.get("hh_size"),
                "vehicles": population.get("vehicles"),
                "commute_mode": population.get("commute_mode"),
                "survey_match_keys": population.get("match_keys"),
                "survey_match_level": population.get("match_level"),
            })

        # No environmental-attitude override: the survey does not measure one.
        # The green weight keeps the base draw from AGENT_DEFAULTS and the
        # Openness-driven shift applied in Agent._refresh_behavior_from_survey,
        # which is the only empirical signal available for it.
        if budget == "low":
            agent.preferences["cost"] = 1.2
        elif budget == "high":
            agent.preferences["cost"] = 0.85
        else:
            agent.preferences["cost"] = 1.0
        agent.preferences["time"] = 1.1 if time_window == "weekday evening" else 0.95
        # travel_affinity and status_seeking keep the base draw from
        # AGENT_DEFAULTS and the shifts applied in
        # Agent._refresh_behavior_from_survey. Their former overrides keyed off
        # travel-party composition and primary leisure interest, neither of
        # which the survey measures; with those held fixed the override was a
        # constant, and a constant attitude carries no information.
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
            # Only the survey-derived adjustment survives. The others keyed off
            # primary leisure interest and travel-party composition, which the
            # survey does not measure and which are now population constants —
            # applying them to every agent alike would shift the whole mixture
            # rather than differentiate anyone.
            ma = pm["motivation_adjustments"]
            mw = dict(agent.motivation_weights)
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

    def _resample_persona(self, idx, personas, rng, used_homes):
        """Extend the population past the given set by resampling a whole profile.

        One respondent is drawn with replacement and copied entire. Resampling
        the profile rather than each attribute independently is what keeps an
        agent's attributes jointly consistent: the correlations among personality,
        platform attitudes and demographics are the observed ones, not the product
        of their marginals. The previous implementation drew each column
        independently, which reproduced every marginal exactly and destroyed every
        correlation — with a survey population that structure is the point, so the
        copies are whole.

        The cost is duplication: a run of N agents over a sample of n respondents
        reuses each profile about N/n times. Duplicates are not clones, because
        latent taste and the home-work pair are drawn per agent downstream, so two
        agents off the same respondent still differ in geography and in what they
        like. ``PersonaID`` records the source respondent so runs can be clustered
        on it.

        The home is only used on unlayered single-city networks; on a metro graph
        :meth:`_sample_home_work` takes the LODES commute pair instead. It is
        redrawn distinct here so copies do not stack on one node, and left as the
        source gives it when the population file carries no coordinates.
        """
        source = rng.choice(personas)
        resampled = dict(source)
        resampled["PersonaID"] = f"{source.get('PersonaID', 'P')}#{idx:05d}"

        try:
            float(source.get("start_latitude", ""))
            float(source.get("start_longitude", ""))
        except (TypeError, ValueError):
            return resampled

        home = self.road_network.sample_home_latlon(rng)
        for _ in range(25):  # resample to keep homes distinct
            if (round(home[0], 6), round(home[1], 6)) not in used_homes:
                break
            home = self.road_network.sample_home_latlon(rng)
        used_homes.add((round(home[0], 6), round(home[1], 6)))
        resampled["start_latitude"] = f"{home[0]:.6f}"
        resampled["start_longitude"] = f"{home[1]:.6f}"
        return resampled

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

    def _load_osm_poi_catalog(self, poi_csv_path, poi_rows=None):
        """Load real leisure POIs (filtered CSV or datasource rows) onto the network.

        POIs are restricted to the principal-city (core) polygon on two-layer
        metro networks — the sim's leisure supply lives only in the core — and
        to the network's bounds otherwise. Capped per category; rating / review /
        popularity are synthesised (the source lacks them — see
        ``_synth_prominence``). Returns ``[]`` when no POI data is available, so
        the simulation falls back to ``_build_synthetic_osm_catalog``.

        Selection itself lives in :func:`welfare_rs.poi_select.select_catalog_rows`
        because the warm-graph builder must make the *identical* choice — the
        chosen POIs are inserted as graph nodes, and those node ids index the
        precomputed routing matrices.
        """
        rows = poi_rows
        if rows is None:
            if not poi_csv_path:
                return []
            path = Path(poi_csv_path)
            if not path.exists():
                return []
            with path.open(newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))

        selected = select_catalog_rows(self.road_network, rows)

        # Insert each POI as a mid-block graph node (at the projection of its real
        # coordinate onto the nearest street) so routing is granular within a
        # block. The POI keeps its EXACT (lat, lon) from the dataset as its
        # location — routing snaps that coordinate to its own inserted node, and
        # the map shows it at the true building position.
        #
        # On a warmed metro these nodes are already in the pickled graph, so this
        # call is a no-op lookup (add_pois_as_nodes is idempotent per place id)
        # and the routing matrices stay valid. It still does real work for
        # unwarmed / legacy networks.
        self.road_network.add_pois_as_nodes(
            [(place_id, lat, lon) for (place_id, _n, _c, lat, lon) in selected]
        )
        # The same venues on the walk and bike networks (baked in, so a no-op,
        # on a warmed metro).
        for _net in self.active_networks.values():
            _net.add_pois_as_nodes(
                [(place_id, lat, lon) for (place_id, _n, _c, lat, lon) in selected])

        # Taste tags come from the business names, so they are assigned over the
        # whole selection at once: the imputation for unlabelled venues draws
        # from the distribution the labelled ones reveal.
        ids = [(pid or f"poi_{i + 1}", name or cat, cat)
               for i, (pid, name, cat, _lat, _lon) in enumerate(selected)]
        tags_by_id = tastes.assign_place_tags(ids)

        catalog = []
        for (place_id, name, cat, lat, lon), (pid, _n, _c) in zip(selected, ids):
            rating, review_count, popularity = self._synth_prominence(place_id)
            catalog.append(
                Place(
                    place_id=pid,
                    name=name or cat,
                    category=cat,
                    location=(lat, lon),
                    keywords=tuple(params.CATEGORY_KEYWORDS.get(cat, (cat,))),
                    taste_tags=tags_by_id.get(pid, ()),
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
        if self.recommender_config is not None:
            return SingleRecommenderOrchestrator(
                ConfigurableRecommender(
                    catalog=self.place_catalog,
                    config=self.recommender_config,
                    dynamics=self.place_dynamics,
                    coord_distance_km=haversine_km,
                )
            )
        # Legacy four-platform stack, kept for callers that predate the builder.
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
                # Core-only on metro networks: synthetic POIs are stand-ins for
                # the principal city's leisure supply.
                location = self.road_network.sample_core_latlon(self.rng)
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
        # Synthetic names carry no taste signal, so every place here is imputed
        # (uniformly over the family's vocabulary — see assign_place_tags).
        tags_by_id = tastes.assign_place_tags(
            [(p.place_id, p.name, p.category) for p in catalog]
        )
        return [
            Place(
                place_id=p.place_id, name=p.name, category=p.category,
                location=p.location, keywords=p.keywords,
                taste_tags=tags_by_id.get(p.place_id, ()),
                rating=p.rating, review_count=p.review_count, popularity=p.popularity,
            )
            for p in catalog
        ]

    def _build_place_catalog(self):
        loaded = self._load_osm_poi_catalog(self.poi_csv_path, self.poi_rows)
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

    # ── Who may use which mode ───────────────────────────────────────────────
    #
    # Car:   only for car owners. The car starts the day at home and goes
    #        wherever the owner drives it; once driven away it is the only mode
    #        until it is back home ("the car can only be used where the car is").
    # Taxi:  only for agents WITHOUT a car: everyone carless in the core, at the
    #        metro's metered fare; carless agents outside the core only in
    #        MODE_AVAILABILITY.outer_taxi_metros, at a share of the fare. Never
    #        tied to the agent, so it needs no return.
    # Walk/  only for agents who live in the core, only between points their
    # bike:  mode's network connects, within the physical limits below. Bike is
    #        bike-share, so it is never tied to the agent and needs no return.

    @staticmethod
    def _owns_car(agent):
        return agent.car_access_type in _CHAIN_BOUND_CAR

    def _taxi_fare_share(self, agent):
        """Share of the metered fare this agent pays for a taxi, or None when
        no taxi is offered to it (it owns a car, or lives outside the core
        outside ``outer_taxi_metros``)."""
        if "taxi" not in self.modes or self._owns_car(agent):
            return None
        if agent.lives_in_core:
            return 1.0
        ma = params.MODE_AVAILABILITY
        if self.metro in ma["outer_taxi_metros"]:
            return float(ma["outer_taxi_fare_share"])
        return None

    def _taxi_fare(self, km, share):
        """Metered fare in dollars for ``km`` of travel (TAXI_FARES), times the
        share of the fare the rider pays."""
        tariff = params.TAXI_FARES.get(self.metro) or params.TAXI_FARES["default"]
        miles = km / 1.609344
        fare = tariff["flag"] + tariff["per_mile"] * miles
        first = tariff.get("first_mile_per_mile")
        if first is not None:
            fare += (first - tariff["per_mile"]) * min(miles, 1.0)
        return share * fare

    @staticmethod
    def _active_permitted(mode, age, ada, km):
        """Physical feasibility of walking or cycling ``km`` (network + access)."""
        ma = params.MODE_AVAILABILITY
        if mode == "walk":
            if age > ma["walk_max_age"]:
                return False
            return km <= (ma["walk_ada_max_km"] if ada else ma["walk_max_distance_km"])
        if mode == "bike":
            if age > ma["bike_max_age"] or ada:
                return False
            return km <= ma["bike_max_distance_km"]
        return False

    @staticmethod
    def _true_point(agent, location):
        """The unsnapped point behind a location: block points for home and
        work, the dataset coordinate (already the location) for a POI."""
        if location == agent.home:
            return agent.home_true
        if location == agent.work:
            return agent.work_true
        return location

    @staticmethod
    def _drive_access_km(agent, location):
        """Drive-graph snap gap at a location. Resolved by LOCATION, not by
        activity type: a remote-work day's "work" activity sits at home."""
        if location == agent.home:
            return agent.home_access_km
        if location == agent.work:
            return agent.work_access_km
        return 0.0

    def _active_access_km(self, agent, mode, location):
        """Snap gap on ``mode``'s network at a home or work location. POIs are
        inserted into every network as mid-block nodes and carry none, the same
        convention the drive graph uses."""
        if location == agent.home:
            key, point = "home", agent.home_true
        elif location == agent.work:
            key, point = "work", agent.work_true
        else:
            return 0.0
        gap = agent.active_access_km.get((mode, key))
        if gap is None:
            gap = self.active_networks[mode].snap_gap_km(*point)
            agent.active_access_km[(mode, key)] = gap
        return gap

    def _active_leg(self, agent, mode, origin, destination):
        """``{distance_km, access_km}`` for walking or cycling a leg, or None
        when the mode is not offered for it."""
        net = self.active_networks.get(mode)
        if net is None or not agent.lives_in_core:
            return None
        km = net.route_km_or_none(self._true_point(agent, origin),
                                  self._true_point(agent, destination))
        if km is None:
            return None
        access = (self._active_access_km(agent, mode, origin)
                  + self._active_access_km(agent, mode, destination))
        if not self._active_permitted(mode, agent.characteristics["age"],
                                      agent.mobility_needs == "ADA/wheelchair", km + access):
            return None
        return {"distance_km": km, "access_km": access, "car_route": None}

    def _car_leg(self, agent, origin, destination, drive_km, *, planning):
        """The car option for a leg, or None when the agent cannot drive it."""
        ma = params.MODE_AVAILABILITY
        if not self._owns_car(agent):
            return None
        if not planning and not agent.car_out and origin != agent.home:
            return None  # the car is parked at home
        if (agent.car_access_type == "carshare"
                and drive_km < ma["carshare_min_distance_km"]):
            return None
        return {
            "distance_km": drive_km,
            "access_km": self._drive_access_km(agent, origin) + self._drive_access_km(agent, destination),
            "car_route": self.road_network.route_car_latlon(origin, destination),
        }

    def _taxi_leg(self, agent, origin, destination, drive_km):
        """The taxi option for a leg, or None when no taxi is offered.

        Same route as a car (the taxi comes to the door, so the drive-graph
        access gap is driven too); priced by :meth:`_taxi_fare`. No minimum
        distance: the fare and the pickup wait are what make a short taxi ride
        unattractive.
        """
        share = self._taxi_fare_share(agent)
        if share is None:
            return None
        return {
            "distance_km": drive_km,
            "access_km": self._drive_access_km(agent, origin) + self._drive_access_km(agent, destination),
            "car_route": self.road_network.route_car_latlon(origin, destination),
            "fare_share": share,
        }

    def _leg_options(self, agent, origin, destination, *, planning=False):
        """Every mode the agent may use for this leg -> its route inputs."""
        legs = {}
        for mode in self.active_networks:
            leg = self._active_leg(agent, mode, origin, destination)
            if leg is not None:
                legs[mode] = leg
        drive_km = None
        if self.modes.keys() & {"car", "taxi", "transit"}:
            drive_km = self.distance_km(origin, destination)
        if "car" in self.modes:
            leg = self._car_leg(agent, origin, destination, drive_km, planning=planning)
            if leg is not None:
                legs["car"] = leg
        if "taxi" in self.modes:
            leg = self._taxi_leg(agent, origin, destination, drive_km)
            if leg is not None:
                legs["taxi"] = leg
        if "transit" in self.modes:
            # The disabled-by-default stand-in: flat speed over road distance.
            legs["transit"] = {
                "distance_km": drive_km,
                "access_km": self._drive_access_km(agent, origin) + self._drive_access_km(agent, destination),
                "car_route": None,
            }
        if not planning and agent.car_out:
            # The car is here and has to go home with the agent.
            legs = {"car": legs["car"]} if "car" in legs else {}
        return legs

    def _tour_feasible_without_car(self, agent):
        """Whether an owner leaving home now could finish the tour without the
        car: every remaining leg until home needs a walk or bike option.

        Leaving the car at home is only a choice if the rest of the day can be
        done without it — nobody walks to work to find they must drive to the
        restaurant from there. The tour is known in full at this point: the day
        plan fixed the leisure venue this morning.
        """
        sched = agent.schedule
        for j in range(agent.current_activity_index + 1, len(sched) - 1):
            a, b = sched[j].location, sched[j + 1].location
            if a != b and not any(self._active_leg(agent, m, a, b) is not None
                                  for m in self.active_networks):
                return False
            if b == agent.home:
                break
        return True

    # ── Costs ────────────────────────────────────────────────────────────────

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
        if policy == "congestion_pricing" and mode in ("car", "taxi"):
            return cp["congestion_pricing_base"] + cp["congestion_pricing_per_km"] * distance_km
        return 0.0

    def _base_mode_utility(self, agent, mode, leg, road_factor, transit_factor):
        """Time, money, emissions and base utility of one mode over one leg.

        ``leg`` is from :meth:`_leg_options`: the mode's own network distance,
        its un-networked access distance, and (car) the fastest-path route.
        Returns ``(parts, travel_time, monetary_cost, emissions, gen_cost,
        travelled_km)``, where ``parts`` holds the utility contributions of time,
        money, comfort and emissions separately (see :meth:`_mode_outcome`).
        """
        uw = params.UTILITY_WEIGHTS
        spec = self.modes[mode]
        weather = max(1e-3, self._weather_speed_factor(mode))
        distance_km = leg["distance_km"]
        car_route = leg.get("car_route")
        on_road = mode in ("car", "taxi")
        if on_road and car_route is not None:
            # Network fastest-path (free-flow) time — freeway km are faster
            # than surface km, which matters for metro commutes. Weather slows
            # it the same way it scaled the flat speed; congestion multiplies
            # as before. Costs/emissions accrue over the driven path's km.
            base_min, car_km = car_route
            in_vehicle = (base_min / weather) * road_factor
            distance_km = car_km
        else:
            # Walk and bike: shortest path on their own network at the mode's
            # speed. (Car only lands here on a graph without edge times.)
            in_vehicle = (distance_km / (spec["speed_kmh"] * weather)) * 60
            if on_road:
                in_vehicle *= road_factor
        if mode == "transit":
            in_vehicle *= transit_factor

        # Access leg: the un-networked hop between the real block point and the
        # graph node it snapped to. It is off-graph by construction, so it is
        # charged straight-line and its km accrue cost and emissions like any
        # other. A walker or cyclist covers it at their own speed; car at local-
        # street speed (CITY_PARAMS.access_speed_kmh). No congestion factor.
        access_km = leg["access_km"]
        if access_km > 0.0:
            if mode in self.active_networks:
                access_speed = spec["speed_kmh"]
            else:
                access_speed = params.CITY_PARAMS.get("access_speed_kmh") or spec["speed_kmh"]
            in_vehicle += (access_km / (access_speed * weather)) * 60
            distance_km += access_km

        wait = spec["wait_min"]
        travel_time = in_vehicle + wait

        if mode == "taxi":
            # The meter runs over the whole driven distance, access included.
            monetary_cost = self._taxi_fare(distance_km, leg.get("fare_share", 1.0))
        else:
            monetary_cost = spec["fixed_cost"] + spec["cost_per_km"] * distance_km
        # Time-based fares (bike-share): minutes beyond the included ride time.
        overage = spec.get("overage_per_min", 0.0)
        if overage:
            monetary_cost += overage * max(0.0, in_vehicle - spec.get("included_min", 0.0))
        monetary_cost += self._policy_cost_adjustment(mode, distance_km)

        gen_cost = (travel_time / 60) * agent.vot * agent.preferences["time"]
        gen_cost += monetary_cost * agent.preferences["cost"]

        emissions = spec["emissions_g_per_km"] * distance_km
        comfort = spec["comfort"]
        green_weight = agent.preferences["green"]
        if self.context["social_norms"] == "green":
            green_weight *= uw["green_norm_boost"]

        denom = uw["gen_cost_denominator"]
        parts = {
            "time": -(travel_time / 60) * agent.vot * agent.preferences["time"] / denom,
            "money": -monetary_cost * agent.preferences["cost"] / denom,
            "comfort": comfort * agent.preferences["comfort"],
            "emissions": -green_weight * (emissions / uw["emissions_divisor"]),
        }
        return parts, travel_time, monetary_cost, emissions, gen_cost, distance_km

    def _leisure_mode_adjustment(self, mode, next_activity):
        if next_activity.type != "leisure" or not next_activity.subtype:
            return 0.0
        # A taxi is a car trip: it takes car's leisure-type taste shift.
        key = "car" if mode == "taxi" else mode
        return params.LEISURE_MODE_TASTE_SHIFTS.get(next_activity.subtype, {}).get(key, 0.0)

    def _mode_outcome(self, agent, mode, leg, road_factor, transit_factor, peer_status,
                      next_activity, with_activity=True):
        """Utility/cost outcome for a single mode over one leg.

        Travel utility is kept in two groups, because the regret rule treats
        them differently (Chorus 2010):

        * ``attributes`` — properties of the mode on this trip: time, money,
          comfort, emissions, the time-based MTTC terms (derived-demand penalty,
          enjoyment of travel), escape and status. Regret agents compare these
          mode against mode.
        * ``constants`` — the agent's standing leaning towards a mode: the
          calibrated constant, community norm, survey-measured mode attitude,
          leisure-type taste shift and the no-car penalty. They are added as
          they are, as alternative-specific constants are in random regret
          models.

        Travel utility is the sum of both, for every decision rule.
        """
        uw = params.UTILITY_WEIGHTS
        attrs, travel_time, monetary_cost, emissions, gen_cost, travelled_km = self._base_mode_utility(
            agent, mode, leg, road_factor, transit_factor,
        )

        attrs["derived"] = 0.0
        if next_activity.is_mandatory:
            derived = agent.motivation_weights["derived"]
            attrs["derived"] = -uw["derived_penalty_coeff"] * derived * (travel_time / uw["derived_time_divisor"])

        intrinsic = agent.motivation_weights["intrinsic"]
        attrs["enjoyment"] = intrinsic * self.mode_enjoyment[mode] * (travel_time / uw["intrinsic_time_divisor"])

        escape = agent.motivation_weights["escape"]
        attrs["escape"] = 0.0
        constants = {"leisure_shift": 0.0}
        if next_activity.type == "leisure":
            # Distance of the route itself (the mode's own network), not the
            # access hop — the same quantity as before modes had networks.
            attrs["escape"] = escape * min(uw["escape_cap"], leg["distance_km"] / uw["escape_distance_scale_km"])
            constants["leisure_shift"] = self._leisure_mode_adjustment(mode, next_activity)

        positionality = agent.motivation_weights["positionality"]
        status_delta = self.mode_status[mode] - peer_status
        attrs["status"] = positionality * agent.attitudes["status_seeking"] * status_delta

        constants["community"] = self.community_mode_bias(mode) * agent.attitudes["practice_conformity"]
        # The survey-measured attitude towards car applies to taxi as well.
        constants["attitude"] = agent.mode_preference_bias.get("car" if mode == "taxi" else mode, 0.0)
        constants["asc"] = self._mode_asc.get(mode, 0.0)
        constants["no_car"] = -agent.car_access_penalty if mode == "car" else 0.0

        travel_utility = sum(attrs.values()) + sum(constants.values())
        activity_utility = 0.0
        if with_activity and next_activity.type == "leisure" and next_activity.subtype:
            seg_cfg = params.LEISURE_SEGMENTS.get(next_activity.subtype, {})
            activity_utility += seg_cfg.get("activity_utility", 0.0)
            activity_utility += uw["activity_intrinsic_coeff"] * agent.motivation_weights["intrinsic"]
            activity_utility += uw["activity_escape_coeff"] * agent.motivation_weights["escape"]
            # Realised taste match at the destination actually reached. This is
            # the term that flows into the trip's activity_utility, hence into
            # leisure_net_utility (the welfare metric) and into the like/dislike
            # the platform learns from.
            activity_utility += self.taste_bonus(agent, next_activity.place_id)
        utility = travel_utility + activity_utility

        return {
            "utility": utility,
            "travel_utility": travel_utility,
            "activity_utility": activity_utility,
            "travel_time": travel_time,
            "cost": monetary_cost,
            "emissions": emissions,
            "gen_cost": gen_cost,
            # Full cost: the whole travel utility in dollars (utility x
            # -gen_cost_denominator). What the satisficing rule measures
            # against its aspiration level, so that it weighs everything the
            # other rules weigh rather than money and time alone.
            "full_cost": -travel_utility * uw["gen_cost_denominator"],
            "attributes": attrs,
            "constants": constants,
            "distance_km": travelled_km,
        }

    def _leg_outcomes(self, agent, origin, destination, next_activity, *, planning=False):
        """Outcome of every mode the agent may use for one leg.

        ``planning`` is the ex-ante view used to price venues: free-flow roads,
        no crowding, yesterday's peer mode mix, and the car assumed available
        to its owner (the morning's mode is not chosen yet).
        """
        legs = self._leg_options(agent, origin, destination, planning=planning)
        if planning:
            road_factor = transit_factor = 1.0
            peer_status = self._planning_peer_status
        else:
            road_factor = self._road_congestion_factor(self.last_road_volume)
            transit_factor = self._transit_crowding_factor(self.last_transit_volume)
            peer_status = self._peer_status_average()
        return {
            mode: self._mode_outcome(agent, mode, leg, road_factor, transit_factor, peer_status,
                                     next_activity, with_activity=not planning)
            for mode, leg in legs.items()
        }

    def _departure_outcomes(self, agent, current_activity, next_activity):
        """Mode outcomes for a departure now, with the car rule applied."""
        origin, destination = current_activity.location, next_activity.location
        outcomes = self._leg_outcomes(agent, origin, destination, next_activity)
        if (self._owns_car(agent) and not agent.car_out and origin == agent.home
                and "car" in outcomes and len(outcomes) > 1
                and not self._tour_feasible_without_car(agent)):
            outcomes = {"car": outcomes["car"]}
        if not outcomes:
            # Unreachable by construction (the population filter and the
            # planner both exclude it); counted so a run can prove it never
            # happens, and priced as a car trip with the agent's no-car
            # penalty so the day can still complete.
            self.infeasible_legs += 1
            legs = {"car": {
                "distance_km": self.distance_km(origin, destination),
                "access_km": self._drive_access_km(agent, origin) + self._drive_access_km(agent, destination),
                "car_route": self.road_network.route_car_latlon(origin, destination),
            }}
            outcomes = {"car": self._mode_outcome(
                agent, "car", legs["car"], self._road_congestion_factor(self.last_road_volume),
                self._transit_crowding_factor(self.last_transit_volume),
                self._peer_status_average(), next_activity)}
        return outcomes

    # ── Planning (the ex-ante price of a venue) ──────────────────────────────

    def _leg_expected_cost(self, agent, origin, destination, stop, gen_cost_weight):
        """Generalised cost the agent can expect on one leg, in utility units,
        or None when no permitted mode makes the leg.

        ``sum_m P_m * C_m / gen_cost_denominator``: each mode's generalised cost
        (time x value of time + money, the paper's C) weighted by the
        probability that THIS agent picks that mode under its own decision
        rule (:meth:`mode_choice_probabilities`, computed from the same full
        utilities the departure will use). ``gen_cost_weight`` is the agent's
        trip-burden multiplier: how heavily it perceives the cost, not which
        mode it expects to take.
        """
        outcomes = self._leg_outcomes(agent, origin, destination, stop, planning=True)
        if not outcomes:
            return None
        denom = params.UTILITY_WEIGHTS["gen_cost_denominator"]
        probs = self.mode_choice_probabilities(agent, outcomes)
        return gen_cost_weight * sum(p * outcomes[m]["gen_cost"] for m, p in probs.items()) / denom

    def planned_round_trip_disutility(self, agent, origin, destination, subtype,
                                      gen_cost_weight=1.0):
        """Expected travel cost of going to ``destination`` and back home, in
        utility units. None when a leg has no permitted mode.

        This is the paper's C in U = V - C: the generalised cost (time x value
        of time + money) of reaching and returning from the activity, which is
        also exactly what the welfare metric charges a realised trip
        (``activity_utility - gen_cost / gen_cost_denominator``). It is made
        mode-aware by weighting each mode's cost by the probability this agent
        takes it: a carless core resident pays the walk, a suburban driver the
        drive.

        Why cost and not the whole travel utility: the travel utility also
        carries the mode constants, which a choice model identifies only
        RELATIVE to car (car = 0). Their level is arbitrary, and priced as a
        level they made every non-car option look absolutely worse: NYC leisure
        participation fell from 0.276 to 0.060. They still decide WHICH mode the
        agent expects to take, through the probabilities. And why not the
        logsum: at this model's utility scale its choice-value term (up to ln 3
        ~ 1.1 per leg) is as large as the travel cost itself, and no realised
        trip is ever credited with it. See Agent._price_round_trip.
        """
        out = self._leg_expected_cost(agent, origin, destination,
                                      _Stop("leisure", subtype, False), gen_cost_weight)
        if out is None:
            return None
        back = self._leg_expected_cost(agent, destination, agent.home,
                                       _Stop("home", "", True), gen_cost_weight)
        if back is None:
            return None
        return out + back

    def reach_limit_km(self, agent):
        """Straight-line bound on how far the agent can travel at all, or None
        when a car or taxi is available (no bound). Network distance is never
        shorter than straight-line, so filtering by this loses no reachable
        venue."""
        if self._owns_car(agent) or self._taxi_fare_share(agent) is not None:
            return None
        if not agent.lives_in_core:
            return 0.0
        ma = params.MODE_AVAILABILITY
        age = agent.characteristics["age"]
        ada = agent.mobility_needs == "ADA/wheelchair"
        limits = [ma["walk_ada_max_km"] if ada else ma["walk_max_distance_km"]] \
            if age <= ma["walk_max_age"] else []
        if age <= ma["bike_max_age"] and not ada:
            limits.append(ma["bike_max_distance_km"])
        return max(limits) if limits else 0.0

    def _graph_counties(self):
        """County FIPS the drive graph serves, or None when it is unlayered.

        The graph's county list minus the metro's excluded counties — the same
        set its commute pairs are restricted to (metro._attach_commutes). The
        exclusions are applied here too because older warm graphs recorded the
        county list before those exclusions existed.
        """
        meta = getattr(self.road_network, "county_meta", None)
        if not meta:
            return None
        excluded = set(params.METRO_PARAMS["metros"].get(self.metro, {}).get("exclude_counties", ()))
        return set(meta) - excluded

    def _population_row_eligible(self, row):
        """Whether a population row's person has a permitted way to work.

        Only carless people can fail. With the taxi, everyone carless in the
        core can always travel, and so can carless people outside the core in
        ``outer_taxi_metros``; carless people outside the core anywhere else
        have no mode at all. The walk/bike check below only matters when the
        taxi mode is disabled. A carless core resident with no job is kept even
        if they can reach nowhere — they have no mandatory trip.
        """
        vehicles = row.get("vehicles")
        if vehicles is None or vehicles > 0:
            return True
        home = (float(row["home_lat"]), float(row["home_lon"]))
        in_core = self.road_network.in_core(*home)
        if "taxi" in self.modes and (
                in_core or self.metro in params.MODE_AVAILABILITY["outer_taxi_metros"]):
            return True
        if not self.active_networks or not in_core:
            return False
        if not row.get("employed") or row.get("work_lat") is None:
            return True
        work = (float(row["work_lat"]), float(row["work_lon"]))
        age = int(row["age"])
        ada = bool(row.get("ambulatory"))
        for mode, net in self.active_networks.items():
            gaps = net.snap_gap_km(*home) + net.snap_gap_km(*work)
            there = net.route_km_or_none(home, work)
            back = net.route_km_or_none(work, home)
            if (there is not None and back is not None
                    and self._active_permitted(mode, age, ada, there + gaps)
                    and self._active_permitted(mode, age, ada, back + gaps)):
                return True
        return False

    # ── Decision paradigms ───────────────────────────────────────────────────

    def mode_choice_probabilities(self, agent, outcomes):
        """Probability of each mode under the agent's own decision rule.

        Mirrors :meth:`choose_mode` exactly, without drawing: the logit for the
        utility paradigm (and as every rule's fallback), random-regret
        probabilities for regret, the habitual mode when available, the
        deterministic prospect pick, and for satisficing the habitual mode first
        and every order of the rest equally likely. Used to price venues ahead of time and by the mode-constant
        calibration.
        """
        if len(outcomes) == 1:
            return {next(iter(outcomes)): 1.0}
        paradigm = agent.decision_paradigms.get("mode_choice", "utility")
        if paradigm == "regret":
            return self._regret_probabilities(outcomes)
        if paradigm == "prospect":
            return {self._choose_mode_prospect(outcomes, agent): 1.0}
        if paradigm == "habit" and agent.habit_mode in outcomes:
            return {agent.habit_mode: 1.0}
        if paradigm == "satisficing":
            head = [agent.habit_mode] if agent.habit_mode in outcomes else []
            rest = [m for m in outcomes if m not in head]
            orders = [head + list(p) for p in itertools.permutations(rest)]
            probs = {m: 0.0 for m in outcomes}
            for order in orders:
                pick = next((m for m in order
                             if outcomes[m]["full_cost"] <= agent.satisficing_threshold), None)
                if pick is None:
                    for m, p in self._logit_probabilities(outcomes).items():
                        probs[m] += p / len(orders)
                else:
                    probs[pick] += 1.0 / len(orders)
            return probs
        return self._logit_probabilities(outcomes)

    @staticmethod
    def _logit_probabilities(outcomes):
        modes = list(outcomes)
        return dict(zip(modes, softmax([outcomes[m]["utility"] for m in modes])))

    def _choose_mode_utility(self, outcomes):
        modes = list(outcomes.keys())
        utilities = [outcomes[m]["utility"] for m in modes]
        probs = softmax(utilities)
        return self.rng.choices(modes, weights=probs, k=1)[0]

    @staticmethod
    def _rrm_probabilities(outcomes):
        """Random regret minimisation (Chorus 2010, *EJTIR* 10(2)).

        Each mode is compared with every other mode attribute by attribute;
        regret builds up whenever another mode is better on an attribute:

            R_i = sum_{j != i} sum_m ln(1 + exp(a_jm - a_im))

        where ``a_im`` is attribute m's contribution to mode i's travel utility
        (see :meth:`_mode_outcome`), so the taste weights are the same ones every
        other rule uses. Alternative-specific constants are not compared; they
        enter additively, as in RRM applications. Choice is a logit over
        negative regret plus constants, the random-regret counterpart of the
        utility agents' logit.

        Regret on TOTAL utility (max_j U_j - U_i) would choose exactly what a
        utility maximiser chooses; comparing attributes one by one is what makes
        regret a distinct rule (it rewards modes that are never far behind on
        anything, the "compromise effect").
        """
        modes = list(outcomes)

        def softplus(x):
            return x + math.log1p(math.exp(-x)) if x > 0 else math.log1p(math.exp(x))

        scores = []
        for i in modes:
            ai = outcomes[i]["attributes"]
            regret = 0.0
            for j in modes:
                if j == i:
                    continue
                aj = outcomes[j]["attributes"]
                regret += sum(softplus(aj[m] - ai[m]) for m in ai)
            scores.append(-regret + sum(outcomes[i]["constants"].values()))
        return dict(zip(modes, softmax(scores)))

    @staticmethod
    def _utility_gap_regret_pick(outcomes):
        """The predecessor paper's regret (Uğurel & Yabe 2026, Eq. 6 / SI Eq. 4):
        ``R_i = max_j U_j - U_i``, and the option with the lowest regret is
        taken, deterministically. U is the full travel utility.

        Note what this implies: the lowest ``max - U_i`` is always the highest
        ``U_i``, so a regret agent under this rule picks the best mode every
        time — the utility rule without its random term.
        """
        top = max(o["travel_utility"] for o in outcomes.values())
        regret = {m: top - o["travel_utility"] for m, o in outcomes.items()}
        return min(regret, key=regret.get)

    def _regret_probabilities(self, outcomes):
        """Choice probabilities of the regret paradigm under the rule chosen in
        ``params.DECISION_RULE_PARAMS['regret_rule']``."""
        rule = params.DECISION_RULE_PARAMS["regret_rule"]
        if rule == "chorus_rrm":
            return self._rrm_probabilities(outcomes)
        if rule == "max_utility_gap":
            return {self._utility_gap_regret_pick(outcomes): 1.0}
        raise ValueError(f"unknown regret_rule {rule!r}")

    def _choose_mode_regret(self, outcomes):
        probs = self._regret_probabilities(outcomes)
        modes = list(probs)
        if len(modes) == 1:
            return modes[0]
        return self.rng.choices(modes, weights=[probs[m] for m in modes], k=1)[0]

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
        """First acceptable mode in the agent's search order: sequential search
        that stops at the first option clearing a reservation level (Simon 1955;
        Caplin, Dean & Martin 2011, *AER* 101(7)).

        "Acceptable" is judged on FULL cost — the mode's whole travel utility in
        dollars — against the agent's aspiration threshold, so a satisficer
        weighs the same things every other rule weighs. (Judging on money and
        time alone made a $0 bike-share ride acceptable whatever else it cost.)
        The search order is the habitual mode first, then the rest at random:
        a fixed order would decide the outcome, and MODE_PARAMS lists walk first.
        """
        order = list(outcomes)
        self.rng.shuffle(order)
        if agent.habit_mode in outcomes:
            order.remove(agent.habit_mode)
            order.insert(0, agent.habit_mode)
        for mode in order:
            if outcomes[mode]["full_cost"] <= agent.satisficing_threshold:
                return mode
        return self._choose_mode_utility(outcomes)

    def choose_mode(self, agent, current_activity, next_activity):
        """Choose the mode for a departure now. Returns ``(mode, outcome)``."""
        outcomes = self._departure_outcomes(agent, current_activity, next_activity)
        if len(outcomes) == 1:
            mode = next(iter(outcomes))
            return mode, outcomes[mode]
        paradigm = agent.decision_paradigms.get("mode_choice", "utility")
        if paradigm == "regret":
            mode = self._choose_mode_regret(outcomes)
        elif paradigm == "prospect":
            mode = self._choose_mode_prospect(outcomes, agent)
        elif paradigm == "satisficing":
            mode = self._choose_mode_satisficing(outcomes, agent)
        elif paradigm == "habit" and agent.habit_mode in outcomes:
            mode = agent.habit_mode
        else:
            mode = self._choose_mode_utility(outcomes)
        return mode, outcomes[mode]

    # ── Visualisation ────────────────────────────────────────────────────────

    def trip_geometry(self, agent, trip):
        """Route polyline of a realised trip, on the network its mode used."""
        net = self.active_networks.get(trip.mode)
        if net is None:
            return self.road_network.route_geometry_latlon(trip.origin, trip.destination)
        return net.route_geometry_latlon(self._true_point(agent, trip.origin),
                                         self._true_point(agent, trip.destination))

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
        infeasible_before = self.infeasible_legs
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
                    mode, data = self.choose_mode(agent, current_activity, next_activity)
                    chosen.append((agent, current_activity, next_activity, mode, data))

                road_volume = sum(1 for c in chosen if c[3] in ("car", "taxi"))
                transit_volume = sum(1 for c in chosen if c[3] == "transit")

                alpha = sd["reference_point_smoothing"]
                for agent, current_activity, next_activity, mode, data in chosen:
                    # The car goes where its owner drives it, and comes home
                    # with them.
                    if self._owns_car(agent):
                        agent.car_out = (mode == "car") and (next_activity.location != agent.home)
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
                self.last_mode_counts = Counter(c[3] for c in chosen)

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
        stats["infeasible_legs"] = self.infeasible_legs - infeasible_before
        # Tomorrow's plans expect today's mix of modes among one's peers.
        day_trips = sum(stats["mode_counts"].values())
        if day_trips:
            self._planning_peer_status = sum(
                self.mode_status[m] * c for m, c in stats["mode_counts"].items()) / day_trips
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
            "infeasible_legs": stats.get("infeasible_legs", 0),
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

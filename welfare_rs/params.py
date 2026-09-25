"""Central parameter registry for the travel-behaviour ABM.

Every tunable numeric assumption lives here so it can be inspected,
modified at runtime, or swept without touching model logic.

Usage
-----
    from welfare_rs import params, Simulation
    params.MODE_PARAMS["car"]["speed_kmh"] = 25   # patch at runtime
    sim = Simulation(...)                           # picks up the new value
"""

from __future__ import annotations

import os

# Repo root = parent of the welfare_rs package dir. Data files (persona CSVs)
# and the default network/POI cache live there, so paths stay stable now that
# this module lives inside the package. Override the cache with $WELFARE_RS_CACHE.
_REPO_ROOT: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Simplification toggles ──────────────────────────────────────────────────
#
# Each toggle below disables one of the richer modelling assumptions so that
# we can quantify its marginal contribution to the final outcomes. All toggles
# default to ``False`` (i.e., full-rich model). Flip to ``True`` to ablate.
#
# ``CAR_ONLY_MODE`` used to live here. It was removed when walking and cycling
# got their own networks (welfare_rs.active_modes): mode choice is now always
# on. The last commit that could run the car-only model is tagged
# ``car-only-final``.
#
# * ``DERIVED_DEMAND_ONLY``: removes the travel-motivation decomposition.
#   Every agent's motivation weights are forced to
#   ``{derived: 1.0, intrinsic: 0.0, escape: 0.0, positionality: 0.0}`` so
#   that travel is treated purely as derived demand.

SIMPLIFICATION_TOGGLES: dict = {
    # Defaults are ON: the simplified model is the one we now run by
    # default. Flip any of these to ``False`` to restore the
    # corresponding piece of the original (richer) model for comparison.
    # OFF: the MTTC motivation mixture is what makes the agents MTTC agents
    # rather than pure derived demand, and the paper's behavioural-theory claim
    # rests on it. With this on, every agent held derived=1.0 and the other
    # three motivations at exactly 0.0, so the four sites that read them
    # (Agent._tpb_intention, the participation boost, the activity benefit and
    # the feedback score) were all multiplying by zero.
    "DERIVED_DEMAND_ONLY": False,
}

# Willingness-to-accept (eta) model: a baseline from the trust-in-platforms and
# autonomy-preference latent variables, a quality term from the RS's own score,
# and a memory term from the count of previously accepted recommendations.
# This is the only eta model (the richer persona/context-driven variant was
# removed; see Appendix B.4 of ``latex/main.tex`` for the formulation).
SIMPLIFIED_ETA_PARAMS: dict = {
    # eta_base = intercept + trust_coeff*(trust-0.5) - autonomy_coeff*(autonomy-0.5)
    "baseline_intercept": 0.52,
    "baseline_trust_coeff": 0.40,
    "baseline_autonomy_coeff": 0.40,
    # Quality term: delta_qual = quality_weight * (score - 0.5)
    "quality_weight": 0.18,
    # Extra-travel term: delta_cost = -beta_delta_cost * (C_offered - C_own),
    # both priced by Agent._price_round_trip. This is Eq. 2's -beta_D * dC_it:
    # how much further the offered venue is than the one the agent had already
    # chosen for itself, which platforms display beside every result.
    #
    # SIGNED, not clipped. A proximity-led ranker that offers something closer
    # than the agent's own pick earns an acceptance bonus, which is the paper's
    # "an agent about to walk 3 blocks can decide instead to walk an extra
    # block" read in both directions.
    #
    # Value: the paper gives no number for beta_D, and no study estimates a
    # distance elasticity of recommendation acceptance, so this is specified
    # rather than estimated — the same standing as the rest of the ICLV map.
    # It is set against a design target: the gap in mean dC between a
    # proximity-led and a footfall-led ranker (Manhattan, 400 agents, 6 days)
    # should become a ~0.195 gap in eta, a ~20-point acceptance difference —
    # enough that distance matters, while staying below the trust term's
    # +/-0.20 so it informs adoption rather than dominating it.
    #
    # History: 0.30 when dC was priced at 0.12 per drive km (gap 0.650); 0.22
    # once venues were priced at expected generalised cost across walk, bike
    # and car (gap 0.870). Current value, 2026-09-23, with the taxi mode:
    # mean dC -0.293 proximity-led vs +1.812 footfall-led (gap 2.106, seed 42,
    # ~265 offers per arm) and gap 1.754 at seed 43; pooled gap 1.930 ->
    # 0.195 / 1.930 = 0.101. The gap is driven by a long right tail (p90 dC
    # ~ +4.9): a footfall-led ranker sends carless core residents to far venues
    # they must reach by taxi at metered fares. At this value a typical offer's
    # extra cost moves eta only a little and the expensive tail is what bites.
    # Reproduce with scripts/measure_delta_cost.py nyc 400 6 <seed>.
    "beta_delta_cost": 0.10,
    # Memory term: delta_mem = memory_weight * min(memory_cap, n_accepted)
    "memory_weight": 0.015,
    "memory_cap": 5,
    "eta_min": 0.05,
    "eta_max": 0.95,
}

# ── Paths ────────────────────────────────────────────────────────────────────

DEFAULT_PERSONA_CSV_PATH: str = os.path.join(
    _REPO_ROOT, "data", "synthetic_personas_realism_first_dopt_with_start_locs_Seattle.csv"
)

# Persona file for the OSM / New York frontend (homes sampled on the network).
NYC_PERSONA_CSV_PATH: str = os.path.join(
    _REPO_ROOT, "data", "synthetic_personas_realism_first_dopt_with_start_locs_NewYork.csv"
)

# Survey-grounded population: one persona per respondent of the April 2026
# Prolific stated-preference survey (IRB-FY2026-11354, N=473). Built from
# data_handoff/ by data/build_survey_personas.py. This is the empirical
# population the paper describes; the two synthetic files above predate it and
# are kept so earlier runs remain reproducible.
SURVEY_PERSONA_CSV_PATH: str = os.path.join(
    _REPO_ROOT, "data", "survey_personas_2026.csv"
)

# ACS/PUMS-built agent populations, exported from public.metro_population by
# db/export_population.py. One CSV per (metro, replicate); the engine reads
# them only when no database is configured, so a clone with downloaded data can
# run without Postgres. Gitignored — populations are data, never code.
POPULATION_DIR: str = os.path.join(_REPO_ROOT, "data", "population")

# ── OSM road-network geography (geo.py) ──────────────────────────────────────
#
# City presets define the area to download from OpenStreetMap. The network is
# downloaded once and cached to GraphML under ``cache_dir``; routing is then
# fully local (no per-run API usage). bbox order is (west, south, east, north).

GEO_PARAMS: dict = {
    "cache_dir": os.environ.get(
        "WELFARE_RS_CACHE",
        os.path.join(_REPO_ROOT, ".cache"),
    ),
    # Overpass endpoints for the one-off metro downloads, PROBED AT RUN TIME
    # and ordered fastest-first (see welfare_rs.netfix.pick_overpass_url).
    #
    # Public Overpass instances fail and recover on their own schedule, and a
    # failure aborts a metro build tens of minutes in (metro.py sets
    # use_cache=False, so nothing is retained). Hardcoding one endpoint just
    # moves the outage around: overpass-api.de was serving 504s and connect
    # timeouts, so this was pinned to kumi.systems — which then turned out to
    # be ~35x slower (71 s vs 2 s on the same query) and stalled a download
    # outright, by which time overpass-api.de had recovered. So: measure, don't
    # guess.
    #
    # Only FULL-PLANET instances belong here. A regional one such as
    # overpass.osm.ch answers HTTP 200 with ZERO elements outside its region,
    # which would silently produce an empty graph rather than an error.
    #
    # FORMAT: BASE url, with NO trailing "/interpreter" — osmnx appends that
    # itself (its own default is "https://overpass-api.de/api"). Including it
    # here produces ".../api/interpreter/interpreter", which does not resolve
    # and surfaces as a *connect timeout* rather than a 404, so it looks
    # exactly like the upstream outage it is not.
    #
    # ORDER: unmetered mirrors first. overpass-api.de advertises "Rate limit: 2"
    # — two slots per client IP — and a metro build issues far more queries than
    # that over an hour. On 2026-07-31 it warmed Miami fine and then stopped
    # answering our SYNs entirely, on BOTH backend IPs at once, for the whole DC
    # build: that is an IP-level block, not an outage, and it presents as a
    # connect timeout with no 429 to warn you. kumi.systems advertises
    # "Rate limit: 0" (no per-client slot accounting) and kept answering in
    # 0.5 s throughout, so bulk warming belongs there. It is the slower mirror
    # per query, which is the right trade for an unattended overnight job.
    #
    # By 2026-07-31 evening BOTH of those were unusable from this host:
    # overpass-api.de blocked (SYNs dropped on both backends) and kumi timing
    # out on real queries while still accepting TCP. The OSM France and mail.ru
    # instances were verified serving identical, correct US data from here —
    # 71,037 ways for the same Manhattan box on both — so they lead now.
    #
    # NOT in this list, deliberately:
    #   overpass.osm.ch          Switzerland only. Answers US queries with
    #                            HTTP 200 and a count of ZERO — it looks
    #                            healthy and yields empty graphs.
    #   overpass.openstreetmap.fr  Whitelist-only. A small probe query
    #                            succeeds, then the real download is refused:
    #                            "403 Forbidden This service is only available
    #                            to white-listed usages". We are not
    #                            whitelisted, so it is not ours to use.
    #   overpass.private.coffee  Resolves to kumi's IP — same box, so listing
    #                            it adds no redundancy.
    # netfix._probe rejects zero-count responses, so a regional mirror cannot be
    # selected even if one is re-added here by mistake. It cannot detect the
    # whitelist case, because the probe is exactly the kind of small query such
    # instances still allow.
    #
    # $WELFARE_RS_OVERPASS_URL pins one explicitly and skips probing.
    "overpass_urls": (
        "https://maps.mail.ru/osm/tools/overpass/api",
        "https://overpass.kumi.systems/api",
        "https://overpass-api.de/api",
    ),
    "overpass_url": os.environ.get("WELFARE_RS_OVERPASS_URL") or None,
    "default_city": "nyc_manhattan",
    "default_network_type": "drive",
    "cities": {
        # bbox = (west, south, east, north)
        "nyc_manhattan": {"bbox": (-74.020, 40.700, -73.930, 40.820)},
        "nyc_lower_manhattan": {"bbox": (-74.020, 40.700, -73.970, 40.745)},
        "brooklyn_full": {"bbox": (-74.050, 40.565, -73.830, 40.740)},
        "brooklyn_downtown_park_slope": {"bbox": (-74.010, 40.660, -73.955, 40.700)},
        "brooklyn_south_prospect_bay_ridge": {"bbox": (-74.045, 40.575, -73.945, 40.670)},
    },
}

# ── Two-layer metro geography (metro.py) ─────────────────────────────────────
#
# Each metro is ONE road network thought of as two layers: the principal city
# ("core", full drive detail — the only place POIs and work locations live) and
# an arterial-only shell over the metro counties (motorway/trunk/primary/
# secondary), where agent homes may also live. The counties included are those
# composing ``county_coverage`` (90%) of the workers commuting into the core,
# largest first — county flow data itself lives ON DISK (never in git; see
# ``datasource.py``), keyed by the metro keys below.
#
# ``core_places`` are Nominatim geocode queries; their union is the core
# polygon (POI bound + work-location pool + full-detail graph extent).

METRO_PARAMS: dict = {
    # Disk registry of county flows (gitignored; schema in datasource.py).
    "county_registry_path": os.environ.get(
        "WELFARE_RS_METRO_COUNTIES",
        os.path.join(_REPO_ROOT, "data", "metro", "metro_counties.json"),
    ),
    # Keep the smallest prefix of counties (sorted by inbound workers, desc)
    # whose cumulative share of inbound workers reaches this coverage.
    "county_coverage": 0.90,
    # OSM ``highway`` classes kept outside the core (the metro "shell").
    "arterial_filter": (
        '["highway"~"motorway|motorway_link|trunk|trunk_link'
        '|primary|primary_link|secondary|secondary_link"]'
    ),
    "default_metro": "nyc",
    "metros": {
        "nyc": {"label": "New York (Manhattan)",
                "core_places": ["Manhattan, New York, USA"]},
        "seattle": {"label": "Seattle",
                    "core_places": ["Seattle, Washington, USA"],
                    # Clark County (Vancouver WA, on the Portland border) ranks
                    # 5th by inbound volume so ``county_coverage`` pulls it in,
                    # but it is ~270 km south with Thurston/Lewis/Cowlitz —
                    # none of them selected — in between. Its roads download as
                    # an island, the strongly-connected-component cut deletes
                    # them, and its 8,074 "commuters" then snap ~120 km north
                    # into Pierce County and get charged the trip. They are a
                    # LODES reporting artifact anyway: employers book remote and
                    # multi-site staff to a headquarters block, so nobody is
                    # really driving Vancouver->Seattle daily.
                    "exclude_counties": ("53011",)},
        "sf_bay": {"label": "SF Bay Area (SF + Oakland)",
                   "core_places": ["San Francisco, California, USA",
                                   "Oakland, California, USA"],
                   # Los Angeles (550 km) and San Diego (750 km) counties, with
                   # the entire Central Valley and Central Coast unselected in
                   # between. Same island defect as Seattle's Clark County
                   # below. 46,585 workers, 5.8% of inbound. Sacramento /
                   # San Joaquin / Sonoma are KEPT: each chains to the core
                   # through a selected neighbour (Solano, Contra Costa, Marin).
                   "exclude_counties": ("06037", "06073")},
        "chicago": {"label": "Chicago",
                    "core_places": ["Chicago, Illinois, USA"]},
        "houston": {"label": "Houston",
                    "core_places": ["Houston, Texas, USA"],
                    # The other four Texas metros: San Antonio (Bexar, 300 km),
                    # Austin (Travis + Williamson, 265 km), Dallas (Dallas +
                    # Collin, 380 km) and Fort Worth (Tarrant, 430 km), none of
                    # them adjacent to anything else selected. Beyond the island
                    # defect this is what made the shell polygon span most of
                    # Texas — 57x osmnx's max query area, so ~57 back-to-back
                    # sub-queries, which is the burst that got this host blocked
                    # by overpass-api.de. 80,475 workers, 5.8% of inbound.
                    "exclude_counties": ("48029", "48453", "48113",
                                         "48439", "48491", "48085")},
        "dc": {"label": "Washington, DC",
               "core_places": ["Washington, District of Columbia, USA"]},
        "miami": {"label": "Miami",
                  "core_places": ["Miami, Florida, USA"]},
        "la": {"label": "Los Angeles",
               "core_places": ["Los Angeles, California, USA"]},
    },
}

# ── Walk and bike networks (active_modes.py) ─────────────────────────────────
#
# Core-only OSM networks for the two active modes. See welfare_rs.active_modes
# for why each setting exists; changing any of them changes the cache identity
# and forces a fresh download.

ACTIVE_NETWORK_PARAMS: dict = {
    "modes": ("walk", "bike"),
    # Extent beyond the (hole-filled) core polygon, so routes can use the
    # streets just past the city line and bridges can reach the far bank.
    "buffer_m": 1000.0,
    # Weakly connected pieces smaller than this are dropped. They are parking
    # aisles and isolated park paths; kept, they capture a nearby home's snap
    # and leave it with no route anywhere. Real islands (Roosevelt Island,
    # Treasure Island, Oakland versus San Francisco) are far larger.
    "min_component_nodes": 50,
    # Keep streets whose sidewalks are mapped as separate ways. OSMnx's walk
    # filter drops them, expecting the separate sidewalk to carry the walker,
    # but those sidewalks are often not drawn across bridges, causeways and
    # crossings — in Miami that cut Watson Island off from the mainland on foot
    # while bikes and cars crossed. The street centreline runs alongside its
    # sidewalk, so keeping it restores connectivity at no distance cost.
    "walk_keep_separate_sidewalk_streets": True,
    # Unioned with OSMnx's walk filter: cycleways where walking is explicitly
    # allowed. OSMnx drops every highway=cycleway, but shared-use trails are
    # commonly tagged cycleway + foot=designated — DC alone has ~260 km of them
    # (Mount Vernon Trail, Rock Creek), which walkers were routed around.
    # Untagged cycleways stay excluded: only explicit permission counts, the
    # same rule the bike network applies to footways.
    "walk_extra_filters": (
        '["highway"="cycleway"]["foot"~"^(yes|designated|permissive)$"]'
        '["area"!~"yes"]["access"!~"private"]["service"!~"private"]',
    ),
    # Unioned with OSMnx's own bike filter: footways where cycling is
    # explicitly allowed, which OSMnx's filter drops wholesale.
    "bike_extra_filters": (
        '["highway"="footway"]["bicycle"~"^(yes|designated|permissive)$"]'
        '["area"!~"yes"]["access"!~"private"]["service"!~"private"]',
    ),
    # Add the reverse of one-way streets tagged oneway:bicycle=no (or the older
    # cycleway=opposite*).
    "bike_contraflow": True,
}

# ── Real POI dataset (filtered per metro core city) ──────────────────────────
#
# The raw dataset (poi_children_merged_by_wkt.csv, ~375 MB, all US) is filtered
# once per metro to core-city leisure POIs by ``data/filter_metro_pois.py``
# into ``<cache>/pois/<metro>_leisure_pois.csv`` (local data, never in git);
# ``LocalDataSource.poi_rows`` serves them and ``Simulation._load_osm_poi_
# catalog`` loads them. NAICS / category-text are mapped to the recommender's
# leisure categories (see recommender_systems.LEISURE_SUBTYPE_TO_CATEGORIES).
# The raw file has no ratings/reviews, so those prominence signals are
# synthesised deterministically per place at load time.

# Legacy single-file pointer (the pre-metro NYC filter output).
NYC_POI_CSV_PATH: str = os.path.join(GEO_PARAMS["cache_dir"], "pois", "nyc_leisure_pois.csv")

POI_PARAMS: dict = {
    # Cap POIs kept per category (fixed-seed random sample). Historically 600,
    # chosen so dense areas stayed responsive when every POI was a live routing
    # target. The precomputed routing matrices remove that cost, so the cap is
    # now a modelling choice rather than a performance one. See
    # welfare_rs.poi_select for the (deterministic) sampling.
    "max_per_category": 1000,
    # NAICS 6-digit code -> leisure category understood by the recommenders.
    "naics_to_category": {
        "722511": "restaurant",      # full-service restaurants
        "722513": "food_takeout",    # limited-service restaurants
        "722514": "restaurant",      # cafeterias / buffets
        "722515": "cafe",            # snack & nonalcoholic beverage bars (coffee)
        "722410": "restaurant",      # drinking places (bars) — nightlife proxy
        "712110": "museum",          # museums
        "712120": "museum",          # historical sites
        "712130": "park",            # zoos & botanical gardens
        "712190": "park",            # nature parks & other similar institutions
        "713940": "fitness_studio",  # fitness & recreational sports centres
        "711110": "live_music",      # theater companies & dinner theaters
        "711130": "live_music",      # musical groups & artists
        "711310": "concert_venue",   # promoters of performing arts w/ facilities
    },
    # Fallback substring match on SUB_CATEGORY/TOP_CATEGORY text (most specific
    # first) when no NAICS code maps.
    "text_to_category": [
        ("coffee", "cafe"), ("snack and nonalcoholic", "cafe"),
        ("nature park", "park"), ("botanical", "park"), ("zoos", "park"),
        ("museum", "museum"), ("historical site", "museum"),
        ("fitness", "fitness_studio"), ("recreational sports", "fitness_studio"),
        # Match the NAICS *sub*-category, never the group name. Bare "musical"
        # hit "Sporting Goods, Hobby, Musical Instrument, and Book Stores",
        # labelling every bookstore, toy shop and fabric store a music venue;
        # bare "performing arts" hit all of NAICS 711 incl. spectator sports;
        # bare "theater" hit "Motion Picture Theaters". Keep only places where
        # a live performance actually happens.
        ("musical groups", "live_music"), ("theater companies", "live_music"),
        ("other performing arts", "live_music"), ("promoters", "concert_venue"),
        ("drinking place", "restaurant"), ("full-service restaurant", "restaurant"),
        ("limited-service restaurant", "food_takeout"), ("restaurant", "restaurant"),
    ],
}

# ── Spatial environment / shared context ─────────────────────────────────────

CITY_PARAMS: dict = {
    "road_capacity_multiplier": 1.8,
    "road_capacity_floor": 60,
    "transit_capacity_multiplier": 2.5,
    "transit_capacity_floor": 120,
    # Organic (non-recommended) leisure choice: when an agent picks a place for the
    # chosen subtype itself, nearer real POIs are favoured via exp(-dist/scale_km).
    # This is the proximity term of the paper's self-selected alternative; the RS,
    # by contrast, also weighs quality/popularity/personalization and may steer the
    # agent to a farther place. Smaller = more strongly local.
    "organic_poi_proximity_scale_km": 3.0,
    "community_mode_bias": {
        "walk": 0.05,
        "bike": 0.10,
        "transit": 0.08,
        "car": -0.02,
        "taxi": -0.02,
        "ai_shuttle": 0.02,
    },
    # Un-networked access leg. Agent homes/workplaces are real Census block
    # points snapped to the nearest graph node; outside the core the graph is
    # arterials only, so that snap can move a suburban home 1-3 km. Charging
    # the trip only from the arterial would make every such commute free for
    # its first kilometres. The residual straight-line distance is instead
    # billed as an access leg at local-street speed (it is genuinely off-graph,
    # so there is no route to measure — a haversine approximation is the honest
    # ceiling on what can be known).
    #
    # This speed is for car (and the disabled transit stand-in). Walking and
    # cycling snap to their own networks and always charge the gap at their own
    # speed: a walker does not cover the last 100 m at 25 km/h. Set this to None
    # to charge car access at the car's flat speed instead.
    "access_speed_kmh": 25.0,
    "default_context": {
        "ai_intervention": "none",
        "social_norms": "standard",
        "travel_norm": "neutral",
        "weather": "fair",
        "authority_constraints": {
            "leisure_open": (9 * 60, 23 * 60),
            "leisure_hours_by_subtype": {
                "food_takeout": (11 * 60, 22 * 60),
                "food_dine_in": (11 * 60, 23 * 60),
                "live_music": (18 * 60, 23 * 60 + 30),
                "workout_or_run": (6 * 60, 22 * 60),
                "cafe_friend": (8 * 60, 20 * 60),
                "museum": (10 * 60, 18 * 60),
                "park": (6 * 60, 21 * 60),
            },
        },
    },
}

# ── Leisure segments ─────────────────────────────────────────────────────────

LEISURE_SEGMENTS: dict = {
    "food_takeout": {
        "label": "Food (takeout)",
        "base_weight": 1.0,
        "duration_minmax": (25, 75),
        "delay_minmax": (10, 45),
        "activity_utility": 0.35,
    },
    "food_dine_in": {
        "label": "Food (dine-in)",
        "base_weight": 1.0,
        "duration_minmax": (60, 150),
        "delay_minmax": (20, 75),
        "activity_utility": 0.55,
    },
    "live_music": {
        "label": "Live music",
        "base_weight": 0.55,
        "duration_minmax": (90, 210),
        "delay_minmax": (45, 150),
        "activity_utility": 0.70,
    },
    "workout_or_run": {
        "label": "Workout class or run",
        "base_weight": 0.95,
        "duration_minmax": (40, 110),
        "delay_minmax": (10, 60),
        "activity_utility": 0.60,
    },
    "cafe_friend": {
        "label": "Cafe with friend",
        "base_weight": 0.9,
        "duration_minmax": (45, 135),
        "delay_minmax": (15, 90),
        "activity_utility": 0.55,
    },
    "museum": {
        "label": "Museum",
        "base_weight": 0.45,
        "duration_minmax": (75, 180),
        "delay_minmax": (25, 120),
        "activity_utility": 0.65,
    },
    "park": {
        "label": "Park",
        "base_weight": 0.75,
        "duration_minmax": (40, 140),
        "delay_minmax": (10, 80),
        "activity_utility": 0.50,
    },
}

CATEGORY_KEYWORDS: dict = {
    "food_takeout": ("takeout", "quick", "food"),
    "restaurant": ("restaurant", "dinner", "food"),
    "cafe": ("cafe", "coffee", "friends"),
    "live_music": ("live", "concert", "music"),
    "concert_venue": ("music", "venue", "concert"),
    "music_event": ("music", "event", "live"),
    "fitness_studio": ("fitness", "workout", "class"),
    "gym": ("gym", "strength", "workout"),
    "running_route": ("run", "outdoor", "fitness"),
    "museum": ("museum", "art", "culture"),
    "park": ("park", "green", "outdoor"),
}

# ── Transport mode parameters ────────────────────────────────────────────────

MODE_PARAMS: dict = {
    "walk": {
        "speed_kmh": 4.8,
        "cost_per_km": 0.0,
        "fixed_cost": 0.0,
        "emissions_g_per_km": 0.0,
        "wait_min": 0,
        "comfort": 0.2,
    },
    # Bike-share, available anywhere in the core (no dock locations yet).
    # Priced as an annual member rides: no unlock fee, the first 45 minutes of
    # a classic-bike ride included, then a per-minute overage. The annual fee is
    # sunk, so it does not enter a per-trip choice. Values are Citi Bike's
    # (NYC) 2026 member terms — $0.27/min overage beyond 45 min — used for every
    # metro until per-system fares are added.
    "bike": {
        "speed_kmh": 14.0,
        "cost_per_km": 0.0,
        "fixed_cost": 0.0,
        "included_min": 45.0,
        "overage_per_min": 0.27,
        "emissions_g_per_km": 0.0,
        "wait_min": 0,
        "comfort": 0.4,
    },
    "transit": {
        "speed_kmh": 22.0,
        "cost_per_km": 0.12,
        "fixed_cost": 1.2,
        "emissions_g_per_km": 30.0,
        "wait_min": 7,
        "comfort": 0.45,
    },
    "car": {
        "speed_kmh": 30.0,
        "cost_per_km": 0.28,
        "fixed_cost": 1.0,
        "emissions_g_per_km": 180.0,
        "wait_min": 1,
        "comfort": 0.7,
    },
    # Taxi: a car trip for a household without a car. Drives the road network
    # at car speed under the same congestion, and adds to it. Priced by the
    # metro's metered fare (TAXI_FARES), not per-km operating cost, so
    # fixed_cost and cost_per_km are unused. wait_min is the pickup wait, an
    # assumption (a few minutes in a dense core).
    "taxi": {
        "speed_kmh": 30.0,
        "cost_per_km": 0.0,
        "fixed_cost": 0.0,
        "emissions_g_per_km": 180.0,
        "wait_min": 5,
        "comfort": 0.7,
    },
}

# Metered taxi fares per metro, from each city's regulator, kept simple: a flag
# fall plus one per-mile rate (and, for Miami, its higher first-mile rate).
# Left out: the short distance the flag covers, slow-traffic time charges,
# surcharges, airport fees and tips — so taxi cost is somewhat UNDERSTATED,
# most in slow Manhattan traffic. "default" (NYC) serves networks with no metro.
TAXI_FARES: dict = {
    # NYC TLC standard metered fare: $3.00 + $0.70 per 1/5 mile.
    "nyc":     {"flag": 3.00, "per_mile": 3.50},
    # Chicago BACP, effective 1 July 2026: $3.25 + $0.31 per 1/9 mile.
    "chicago": {"flag": 3.25, "per_mile": 2.79},
    # DC Department of For-Hire Vehicles: $4.00 first 1/8 mile, $2.56/mile.
    "dc":      {"flag": 4.00, "per_mile": 2.56},
    # SFMTA rates of fare (April 2023): $4.15 + $0.65 per 1/5 mile.
    "sf_bay":  {"flag": 4.15, "per_mile": 3.25},
    # King County / City of Seattle, August 2025: $2.60 + $0.30 per 1/9 mile.
    "seattle": {"flag": 2.60, "per_mile": 2.70},
    # Miami-Dade R-700-22: $2.95 first 1/6 mile, $0.85 per 1/6 mile to 1 mile
    # ($5.10/mile), then $0.55 per 1/6 mile ($3.30/mile).
    "miami":   {"flag": 2.95, "per_mile": 3.30, "first_mile_per_mile": 5.10},
    # City of Houston ARA: $2.91 first 1/11 mile (daytime), $2.31/mile.
    "houston": {"flag": 2.91, "per_mile": 2.31},
    # LADOT 2023 schedule: $3.10 + $0.30 per 1/9 mile. (A licensed operator now
    # lists $0.33 per 1/9 mile, $2.97/mile; the current official schedule could
    # not be retrieved, so the published 2023 rate is used.)
    "la":      {"flag": 3.10, "per_mile": 2.70},
    "default": {"flag": 3.00, "per_mile": 3.50},
}

# Modes a Simulation leaves out unless told otherwise. Transit here is a
# stand-in (flat speed over road distance plus a wait), not a transit network,
# so it is not offered until one exists.
DISABLED_MODES: tuple = ("transit",)

MODE_STATUS: dict = {
    "walk": 0.2,
    "bike": 0.3,
    "transit": 0.4,
    "car": 0.9,
    # A taxi is a car trip: same status, enjoyment, norm and weather effects as
    # car. What differs is who may take it (carless agents), what it costs (a
    # metered fare) and the wait for a pickup.
    "taxi": 0.9,
}

MODE_ENJOYMENT: dict = {
    "walk": 0.5,
    "bike": 0.6,
    "transit": 0.3,
    "car": 0.2,
    "taxi": 0.2,
}

# ── Utility function weights ─────────────────────────────────────────────────

UTILITY_WEIGHTS: dict = {
    "gen_cost_denominator": 8.0,
    "derived_penalty_coeff": 0.05,
    "derived_time_divisor": 10.0,
    "intrinsic_time_divisor": 30.0,
    "escape_distance_scale_km": 5.0,
    "escape_cap": 1.5,
    "green_norm_boost": 1.25,
    "emissions_divisor": 1000.0,
    "activity_intrinsic_coeff": 0.30,
    "activity_escape_coeff": 0.15,
}

LEISURE_MODE_TASTE_SHIFTS: dict = {
    "food_takeout": {"bike": 0.05, "walk": 0.03, "car": -0.02},
    "food_dine_in": {"transit": 0.05},
    "live_music": {"transit": 0.10, "walk": -0.03},
    "workout_or_run": {"walk": 0.18, "bike": 0.14, "car": -0.10},
    "cafe_friend": {"walk": 0.10, "bike": 0.07, "car": -0.04},
    "museum": {"transit": 0.09, "walk": 0.06, "car": -0.03},
    "park": {"walk": 0.15, "bike": 0.10, "car": -0.08},
}

# ── Congestion / crowding ────────────────────────────────────────────────────

CONGESTION_PARAMS: dict = {
    "bpr_alpha": 0.15,
    "bpr_beta": 4,
    "transit_crowding_coeff": 0.25,
    "transit_crowding_exponent": 2,
    "congestion_pricing_base": 2.0,
    "congestion_pricing_per_km": 0.15,
}

# ── Weather speed factors ────────────────────────────────────────────────────

WEATHER_FACTORS: dict = {
    "rain": {"bike": 0.75, "walk": 0.75, "transit": 0.9, "car": 0.9, "taxi": 0.9},
    "heat": {"bike": 0.80, "walk": 0.80, "transit": 1.0, "car": 1.0, "taxi": 1.0},
    "fair": {"bike": 1.0, "walk": 1.0, "transit": 1.0, "car": 1.0, "taxi": 1.0},
}

# ── TPB intention model ──────────────────────────────────────────────────────

TPB_PARAMS: dict = {
    "attitude_coeff": 1.3,
    "norm_coeff": 0.9,
    "pbc_coeff": 0.8,
    "intercept": -1.0,
    "leisure_attitude_boost": 0.1,
    "norm_pro_travel": 0.8,
    "norm_neutral": 0.5,
    "norm_anti_travel": 0.2,
    "pbc_own_car": 0.4,
    "pbc_carshare": 0.18,
    "pbc_transit_weight": 0.25,
    "pbc_age_75_penalty": -0.1,
    "pbc_ada_penalty": -0.08,
    "pbc_base": 0.1,
}

# ── Leisure participation / subtype choice ───────────────────────────────────

PARTICIPATION_PARAMS: dict = {
    "signal_intercept": -0.45,
    "signal_intention_coeff": 1.05,
    "signal_motivation_coeff": 0.70,
    "signal_net_utility_coeff": 0.60,
    "motivation_intrinsic_coeff": 0.3,
    "motivation_escape_coeff": 0.2,
    "p_participate_max": 0.95,
    "min_net_utility": -0.15,
    # (expected_trip_disutility_per_km, the flat 0.12/km a venue used to be
    # priced at, is gone: venues are priced by the mode-choice logsum — see
    # Agent._price_round_trip.)
    "outdoor_walk_tol_discount": 0.92,
    "outdoor_walk_tol_threshold": 30,
    "activity_intrinsic_coeff": 0.35,
    "activity_escape_coeff": 0.20,
    "activity_travel_affinity_coeff": 0.15,
    "activity_pref_coeff": 0.25,
    "primary_interest_bonus": 0.12,
    "time_window_bonuses": {
        "weekday evening": {"in_range": 0.08, "out_range": -0.03, "range": (17, 22)},
        "weekend afternoon": {"in_range": 0.05, "out_range": -0.02, "range": (12, 18)},
        "weekend morning": {"in_range": 0.04, "out_range": -0.02, "range": (8, 13)},
    },
    "net_utility_noise": (-0.05, 0.05),
    "softmax_temperature": 2.5,
    "interest_map": {
        "food": {"food_takeout", "food_dine_in", "cafe_friend"},
        "nature": {"park", "workout_or_run"},
        "nightlife": {"live_music", "food_dine_in"},
        "cultural": {"museum", "cafe_friend"},
    },
}

# ── Feedback / RS learning ───────────────────────────────────────────────────

FEEDBACK_PARAMS: dict = {
    "experience_activity_weight": 0.85,
    "experience_travel_weight": 0.25,
    "pref_fit_weight": 0.30,
    "intrinsic_weight": 0.20,
    "escape_weight": 0.10,
    "travel_affinity_weight": 0.15,
    "rain_heat_outdoor_penalty": -0.25,
    "green_norm_park_bonus": 0.10,
    "sigmoid_scale": 1.25,
    "p_like_min": 0.05,
    "p_like_max": 0.95,
    "rs_learning_rate": 0.12,
    "keyword_affinity_discount": 0.70,
    "feedback_strength_floor": 0.25,
    "feedback_strength_max": 2.0,
    "feedback_strength_min": 0.2,
}

# ── Synthetic place catalog ──────────────────────────────────────────────────

CATALOG_PARAMS: dict = {
    "rating_range": (2.8, 4.9),
    "review_count_range": (20, 5000),
    "popularity_multiplier_range": (1.0, 9.0),
    "popularity_floor": 10.0,
    # Synthetic-catalog fallback: POIs are placed on random network nodes, this
    # many (lo, hi) per category, guaranteeing candidates for every subtype.
    "osm_places_per_category": (6, 14),
}

# ── Latent tastes (welfare_rs.tastes) ────────────────────────────────────────
#
# Each POI carries taste tags recovered from its business name ("Joe's Pizza" ->
# pizza); each agent holds a latent favourite per taste family. The tag is NOT
# visible to the recommenders — it moves the agent's own realised activity
# utility, and the platform can only infer it from the like/dislike feedback
# that utility produces. That makes personalization a genuine discovery problem
# rather than a lookup.

TASTE_PARAMS: dict = {
    # Extra activity utility when the visited POI matches a taste in its family.
    # Sized against LEISURE_SEGMENTS' activity_utility (0.35-0.70), so a match is
    # worth roughly a quarter of a typical outing's benefit: enough to move
    # feedback and welfare, not enough to swamp the travel-cost trade-off that
    # the spatial metrics measure.
    "match_bonus": 0.15,
    # Tempering exponent on the supply-derived taste distribution. Weights go as
    # ``share ** beta``: 1.0 tracks the observed venue mix exactly, 0.0 is
    # uniform. Below 1 flattens the head, correcting the two biases that inflate
    # it (self-labelling venues are over-counted; small-format types need more
    # premises per unit demand) and ensuring some agents want something locally
    # scarce — without that, taste never moves anyone anywhere.
    "beta": 0.7,
    # Favourites per family; 0 = scale to the family's variety (~1 per 8 types).
    "per_family": 0,
}

# ── Dynamic POI prominence (PlaceDynamics) ───────────────────────────────────
#
# Live per-POI state accumulated during a run: every leisure visit (recommended
# or organic) adds footfall, and every like/dislike from an accepted
# recommendation posts a "review" that moves the place's rating via a Bayesian
# average around the synthetic base rating.
#
# Dynamic popularity is *pure footfall* — the count of recorded visits. The
# synthetic base popularity on each Place is not carried into it (it only
# matters when no PlaceDynamics is wired), so rankings respond to what agents
# actually do rather than to a fabricated visitor history.

POI_DYNAMICS: dict = {
    # Pseudo-review weight of the base (prior) rating: higher = new feedback
    # moves the public rating more slowly.
    "prior_weight": 20.0,
    # Star value a like / dislike contributes to the rating average.
    "like_star": 5.0,
    "dislike_star": 1.0,
    "rating_min": 1.0,
    "rating_max": 5.0,
    # Flat pseudo-visit count added to every place's footfall. 0 = raw visit
    # counts (strongest rich-get-richer feedback); raising it smooths early
    # dynamics when only a handful of visits have happened.
    "popularity_prior_visits": 0.0,
}

# ── Agent defaults ───────────────────────────────────────────────────────────

AGENT_DEFAULTS: dict = {
    # Random-agent population
    "income_range": (20000, 110000),
    "age_range": (18, 80),
    "car_ownership_prob_low_income": 0.45,
    "car_ownership_prob_high_income": 0.72,
    "car_ownership_income_threshold": 50000,
    "no_car_access_penalty": 2.0,
    # Preference draws
    "pref_time_range": (0.8, 1.2),
    "pref_cost_range": (0.8, 1.2),
    "pref_comfort_range": (0.0, 1.0),
    "pref_green_range": (0.0, 1.0),
    # Attitude draws
    "attitude_range": (0.0, 1.0),
    # Motivation Dirichlet
    "motivation_dirichlet": [2.0, 1.5, 1.2, 1.0],
    # Leisure preferences
    "leisure_pref_range": (0.7, 1.3),
    "feedback_sensitivity_range": (0.8, 1.2),
    "psych_default_range": (0.35, 0.65),
    # Decision paradigm weights
    "paradigm_weights": {
        "utility": 0.45,
        "regret": 0.20,
        "prospect": 0.15,
        "satisficing": 0.10,
        "habit": 0.10,
    },
    # Prospect theory
    "loss_aversion_range": (1.7, 2.7),
    "ref_time_range": (20, 40),
    "ref_cost_range": (2.0, 6.0),
    # Satisficing
    "satisficing_base_range": (6.0, 16.0),
    "satisficing_income_divisor": 100000,
    # Value of time. income / vot_hours_per_year is an hourly wage equivalent;
    # VOT is a fraction of it. Small (2012) puts the value of travel time
    # savings at roughly half the wage rate, which is the employed share here.
    # Non-earners take half again: the survey reports HOUSEHOLD income, so the
    # wage equivalent is not theirs to begin with.
    "vot_floor": 5,
    "vot_hours_per_year": 2000,
    "vot_wage_share_employed": 0.50,
    "vot_wage_share_non_earner": 0.25,
    # Work schedule
    "work_start_range": (7 * 60, 9 * 60),
    "work_duration_range": (7 * 60, 9 * 60),
    # Remote work
    "remote_base": 0.12,
    "remote_no_car_boost": 0.02,
    "remote_age_55_boost": 0.05,
    "remote_cap": 0.35,
}

# ── Survey/ICLV to behavior mapping ─────────────────────────────────────────

SURVEY_BEHAVIOR_PARAMS: dict = {
    "platform_affinity": {
        "trust": 0.30,
        "awareness": 0.25,
        "platform_follow": 0.25,
        "ai_follow": 0.20,
    },
    "autonomy_guard": {
        "autonomy": 0.70,
        "awareness": 0.30,
    },
    "planning_orientation": {
        "conscientiousness": 0.40,
        "maximization": 0.30,
        "low_spontaneity": 0.15,
        # Survey s4_search_c: "I would cross-check options across multiple
        # sources before deciding." A deliberation indicator, same family as
        # conscientiousness and maximization. Weights in this block sum to 1
        # because the result is clamped to [0, 1] and read centred on 0.5.
        "cross_platform_search": 0.15,
    },
    "social_orientation": {
        "extraversion": 0.55,
        "group_coordination": 0.30,
        "social_posting": 0.15,
    },
    "variety_seeking": {
        "openness": 0.55,
        "unexpected_discovery": 0.25,
        "advice_goal_directed": 0.20,
    },
    # Renormalised over the three measured inputs after the survey turned out
    # to carry no disappointing-recommendation item. Weights sum to 1 because
    # the result is clamped to [0, 1] and read centred on 0.5.
    "risk_aversion": {
        "neuroticism": 0.72,
        "autonomy": 0.17,
        "awareness": 0.11,
    },
    "budget_sensitivity": {
        "budget_tightness": 0.60,
        "incentives": 0.40,
    },
    "trend_susceptibility": {
        "herding": 0.55,
        "agreeableness": 0.25,
        "platform_follow": 0.20,
    },
    "ai_affinity": {
        "ai_follow": 0.30,
        "ai_itinerary_comfort": 0.25,
        "trust": 0.20,
        "low_autonomy": 0.25,
    },
    "feedback_loop": {
        "review_posting": 0.25,
        "switch_when_dissatisfied": 0.45,
        "multi_platform_parallel": 0.30,
    },
    "preference_shift": {
        "time_planning": 0.25,
        "time_maximization": 0.15,
        "cost_budget": 0.35,
        "cost_price_filter": 0.20,
        "comfort_risk": 0.28,
        "green_openness": 0.20,
    },
    "preference_bounds": {
        "min": 0.45,
        "max": 1.80,
    },
    "attitude_shift": {
        "travel_affinity_variety": 0.25,
        "travel_affinity_frequency": 0.20,
        # Survey overnight_num: trips per 12 months, a second travel-propensity
        # indicator alongside weekly local outings. These are independent shift
        # magnitudes applied as coeff * (x - 0.5), not weights summing to 1.
        "travel_affinity_overnight": 0.12,
        "status_trend": 0.28,
        "practice_social": 0.20,
        "practice_trend": 0.20,
    },
    "motivation_shift": {
        "derived_planning": 0.20,
        "intrinsic_variety": 0.30,
        "escape_spontaneity": 0.18,
        "positionality_trend": 0.25,
    },
    "feedback_sensitivity_shift": {
        "feedback_loop": 0.35,
        "risk_aversion": 0.20,
    },
    "prospect_shift": {
        "loss_aversion_risk": 0.95,
    },
    "satisficing_shift": {
        "budget": 2.0,
        "maximization": 1.3,
    },
    "paradigm_shift": {
        "utility_variety": 0.45,
        "regret_maximization": 0.85,
        "prospect_risk": 0.90,
        "satisficing_budget": 0.80,
        "satisficing_planning": 0.35,
        "habit_autonomy": 0.75,
        "habit_low_variety": 0.45,
    },
    "mode_bias": {
        "walk_variety": 0.16,
        "walk_budget": 0.12,
        "walk_risk": 0.14,
        "bike_variety": 0.18,
        "bike_green": 0.14,
        "bike_risk": 0.20,
        "transit_budget": 0.18,
        "transit_platform": 0.08,
        "car_autonomy": 0.18,
        "car_risk": 0.12,
        "car_green": 0.22,
    },
    "subtype_utility_shift": {
        "social_subtypes": {"cafe_friend", "food_dine_in", "live_music"},
        "exploration_subtypes": {"museum", "live_music", "park"},
        "planned_subtypes": {"museum", "food_dine_in"},
        "spontaneous_subtypes": {"food_takeout", "park", "workout_or_run"},
        "outdoor_subtypes": {"park", "workout_or_run"},
        "trend_subtypes": {"food_dine_in", "live_music", "cafe_friend"},
        "social_bonus": 0.18,
        "exploration_bonus": 0.20,
        "planned_bonus": 0.12,
        "spontaneous_bonus": 0.12,
        "outdoor_risk_penalty": 0.22,
        "trend_bonus": 0.14,
    },
    "trip_disutility_multiplier": {
        "risk_aversion": 0.35,
        "budget_sensitivity": 0.25,
        "variety_seeking": 0.20,
        "platform_affinity": 0.10,
        "min": 0.65,
        "max": 1.45,
    },
    "feedback_shift": {
        "trust_platforms": 0.16,
        "feedback_loop_strength": 0.12,
        "autonomy_preference": 0.10,
        "awareness_caution": 0.08,
        "feedback_sensitivity_multiplier": 0.25,
    },
    "participation_shift": {
        "social_orientation": 0.18,
        "variety_seeking": 0.18,
        "local_frequency": 0.15,
        "spontaneity": 0.16,
        "city_familiarity": 0.10,
        "softmax_maximization": 0.55,
        "softmax_spontaneity": 0.30,
        "softmax_min": 0.35,
    },
}

# ── Persona agent mapping ────────────────────────────────────────────────────

PERSONA_MAPPING: dict = {
    # Both the synthetic-persona labels and the survey's own band labels are
    # accepted, so one lookup serves either population file.
    #
    # Survey income is HOUSEHOLD income before taxes (Q6), while the value of
    # time it feeds (Agent.vot = income / vot_hours_per_year) is an individual
    # wage equivalent. Multi-earner households therefore get an overstated VOT.
    # The two open-ended bands need an assumed bound: "Under $25,000" is floored
    # at 12000 and "$200,000 or more" capped at 300000.
    "income_bands": {
        "<35k": (20000, 34000),
        "35-75k": (35000, 74000),
        "75k-125k": (75000, 124000),
        "125k+": (125000, 180000),
        # Survey Q6 bands.
        "Under $25,000": (12000, 24999),
        "$25,000 - $49,999": (25000, 49999),
        "$50,000 - $74,999": (50000, 74999),
        "$75,000 - $99,999": (75000, 99999),
        "$100,000 - $149,000": (100000, 149000),
        "$150,000 - $199,999": (150000, 199999),
        "$200,000 or more": (200000, 300000),
    },
    # Survey age is a 6-band ordinal (Q1); years are drawn uniformly inside the
    # band. "65 or older" is capped at 85 to match the synthetic bands' ceiling.
    "age_bands": {
        "teen": (18, 19),
        "young adult": (20, 34),
        "mid adult": (35, 54),
        "old adult": (55, 69),
        "senior": (70, 85),
        # Survey Q1 bands.
        "18-24": (18, 24),
        "25-34": (25, 34),
        "35-44": (35, 44),
        "45-54": (45, 54),
        "55-64": (55, 64),
        "65 or older": (65, 85),
    },
    "transit_access_levels": {"high": 0.85, "medium": 0.55, "low": 0.25},
    # Attributes the survey does not collect. They were previously drawn per
    # agent from the synthetic persona file, whose level shares came from a
    # D-optimal experimental design — balanced by construction (32% ADA needs,
    # 33% carless, 50% non-English) rather than representative of any
    # population. Rather than invent population rates, they are now held fixed
    # for every agent, so they contribute no heterogeneity and cannot be
    # mistaken for an empirical source of it.
    #
    # Car access, transit access and mobility needs are fixed ONLY for agents
    # built without an ACS/PUMS population row. A built population measures all
    # three per agent (PUMS vehicles, tract transit share, PUMS ambulatory
    # difficulty) and overrides these. Primary leisure interest and
    # environmental attitude are dropped outright rather than pinned: both
    # applied a utility bonus, and a constant bonus for all agents is just a
    # shift in the origin. The green weight instead keeps its random base and
    # the survey's Openness shift (SURVEY_BEHAVIOR_PARAMS.preference_shift).
    "fixed_attributes": {
        "car_access": "own car",
        "transit_access": "medium",
        "mobility_needs": "none",
        "walk_tolerance_min": 15,
        # One broad window for everyone. LEISURE_SEGMENTS place most outings in
        # the evening, so the weekday-evening band is the least distorting
        # constant; PARTICIPATION_PARAMS.time_window_bonuses still applies it.
        "time_window": "weekday evening",
        "risk_salience": "low",
        # "" matches no key in PARTICIPATION_PARAMS.interest_map, so no subtype
        # receives the primary-interest bonus.
        "primary_interest": "",
    },
    # Persona columns copied onto Agent.characteristics for reporting and
    # segmentation. Carried only — no decision rule reads them. Absent columns
    # are skipped, so this list is safe against either population file.
    "carried_fields": (
        "Education",
        "Education_num",
        "Employment",
        "RaceEthnicity",
        "HomeLanguage",
        "CensusRegion",
        "CensusDivision",
        "ResidenceLength",
        "ResidenceLength_num",
        "LeisureSpend",
        "LeisureSpend_num",
        "CityFam_num",
        "Age_num",
        "Income_num",
        "Sex_num",
        "Sex",
        "scenario",
        "Autonomy_Control",
        "Platform_Comfort",
    ),
    "no_car_penalty": 2.2,
    "carshare_penalty": 0.9,
    # Motivation adjustment from a persona trait. Only the survey-derived one
    # remains: the others keyed off primary leisure interest and travel-party
    # composition, which the survey does not measure.
    "motivation_adjustments": {
        "top_rated_positionality": 0.14,
    },
}

# ── Decision rules ───────────────────────────────────────────────────────────

DECISION_RULE_PARAMS: dict = {
    # Which regret rule the regret paradigm uses for mode choice. BOTH are kept
    # until the user decides (2026-09-23):
    #   "chorus_rrm"       Chorus (2010) random regret minimisation, cited by the
    #                      GeoRecSim paper: modes compared attribute by
    #                      attribute, R_i = sum_j sum_m ln(1 + exp(a_jm - a_im)),
    #                      constants added, logit over -R. Distinct from utility
    #                      maximisation (it favours all-rounders).
    #   "max_utility_gap"  The predecessor paper's regret (Uğurel & Yabe 2026,
    #                      Eq. 6 / SI Eq. 4, there the recommender's filter):
    #                      R_i = max_j U_j - U_i, lowest taken. Always picks the
    #                      highest-utility mode, i.e. the utility rule without
    #                      its random term.
    # MODE_ASC was calibrated under "chorus_rrm". After switching, re-run
    # scripts/calibrate_mode_constants.py and scripts/measure_delta_cost.py.
    "regret_rule": "chorus_rrm",
}

# ── Mode availability constraints ────────────────────────────────────────────

MODE_AVAILABILITY: dict = {
    "carshare_min_distance_km": 1.5,
    # Taxi is how a household WITHOUT a vehicle travels by car. Every carless
    # agent who lives in the core may take one, at the city's metered fare
    # (TAXI_FARES). Carless agents living OUTSIDE the core may take one only in
    # these metros, at this share of the fare: in NYC 14.7% of agents are
    # carless and live outside Manhattan, where walking and cycling are not
    # offered, and with no transit mode they would otherwise have no way to
    # travel. Half the fare is the user's stand-in for the cheaper options they
    # really use (subway, shared rides). Everywhere else a carless agent living
    # outside the core has no mode and is left out of the population
    # (Simulation._population_row_eligible).
    "outer_taxi_metros": ("nyc",),
    "outer_taxi_fare_share": 0.5,
    # Walking and cycling are offered only to agents who LIVE in the core, and
    # only between points on that mode's network. The distance ceilings below
    # are physical feasibility limits (about an hour's walk, about 50 minutes'
    # ride), not behavioural ones: how far people are willing to walk or ride
    # comes from the time they would spend, priced by their value of time.
    # (The walking limit used to be walk_tolerance_min / 12, i.e. 1.25 km for
    # everyone, because walk_tolerance_min is pinned at 15 — one unsourced
    # constant would have set the walk share.)
    "walk_max_distance_km": 5.0,
    "bike_max_distance_km": 12,
    "bike_max_age": 75,
    "walk_ada_max_km": 1.0,
    "walk_max_age": 85,
}

# PUMS JWTRNS (means of transportation to work) -> the mode an agent's habit
# starts from. Motorcycle is a car trip. Transit (2-6), worked from
# home (11) and other (12) have no counterpart among the modes offered, so
# those agents start with no habit and choose by utility until they have one.
PUMS_JWTRNS_TO_MODE: dict = {1: "car", 7: "taxi", 8: "car", 9: "bike", 10: "walk"}

# Alternative-specific constants per metro, added to a mode's travel utility
# (car and taxi are the reference, 0). Calibrated by
# scripts/calibrate_mode_constants.py (2026-09-23, with the taxi mode and the
# "chorus_rrm" regret rule; 6,000 agents x replicates 0-1 per metro) so that the
# simulated commute walk and bike shares of core-resident workers WHO HAVE A
# CHOICE of mode match those same workers' own PUMS commute modes. Shares are
# exact expected probabilities under each agent's decision rule. Commuters with
# a single permitted mode are excluded (a constant cannot move them); their
# shares are listed as "single-option". A missing metro gets no constants.
#
# The bike constants are large and negative everywhere: they carry what
# deters cycling but is not in the model (safety, weather, effort, finding a
# dock), against a bike-share ride that is otherwise free and fast. Re-run the
# calibration after changing modes, fares, networks or the regret rule.
MODE_ASC: dict = {
    # n = commuters fitted; observed / fitted shares in %.
    "miami":   {"walk": -0.545, "bike": -6.194},  # n=1,333 walk 9.90/9.99 bike 1.28/1.37; single-option car 3.7, taxi 0.4
    "dc":      {"walk": 0.703, "bike": -3.567},   # n=2,633 walk 14.74/14.81 bike 5.09/5.12; single-option car 5.4, taxi 2.2
    "nyc":     {"walk": 0.686, "bike": -5.127},   # n=1,704 walk 28.05/28.14 bike 5.69/5.70; single-option car 3.2, taxi 6.2
    "sf_bay":  {"walk": 0.983, "bike": -3.325},   # n=2,494 walk 16.80/16.72 bike 4.77/4.82; single-option car 18.2, taxi 2.9
    "seattle": {"walk": 0.879, "bike": -3.006},   # n=2,579 walk 13.26/13.23 bike 3.61/3.66; single-option car 17.0, taxi 1.2
    "chicago": {"walk": 0.243, "bike": -5.641},   # n=2,152 walk 9.39/9.41 bike 2.04/2.13; single-option car 34.3, taxi 5.1
    "houston": {"walk": -1.214, "bike": -6.693},  # n=1,144 walk 1.57/1.65 bike 0.35/0.33; single-option car 56.8, taxi 1.9
    "la":      {"walk": -0.624, "bike": -5.482},  # n=1,419 walk 5.07/5.16 bike 0.92/0.93; single-option car 48.5, taxi 2.6
}

# ── Simulation defaults ──────────────────────────────────────────────────────

SIM_DEFAULTS: dict = {
    "num_agents": 200,
    "seed": 42,
    "time_step": 5,
    "reference_point_smoothing": 0.2,   # alpha in exponential smoothing
}

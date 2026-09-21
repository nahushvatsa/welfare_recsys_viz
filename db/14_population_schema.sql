-- The agent population layer: reweighted PUMS, core home blocks, and the
-- built populations the engine reads.
-- Run as welfare_owner:  psql -v ON_ERROR_STOP=1 -f db/14_population_schema.sql

-- ── Home blocks for non-workers: CORE ONLY ───────────────────────────────────
-- Workers get a home from LODES, which spans the whole metro, because their
-- leisure trip departs from their WORKPLACE — and every workplace is in the
-- core, a few km from the venues. A non-worker has no workplace, so its trip
-- departs from home (welfare_rs/agent.py, `origin_after_work`). Drawing those
-- homes metro-wide would put a large share of the population 30-60 km from
-- every POI in the catalog, across a shell layer that carries arterials only.
-- Those trips would dominate every distance-normalised metric while describing
-- a behaviour the model does not actually represent: a suburban non-worker's
-- real leisure options are suburban venues, which are not in the catalog.
--
-- So the non-worker population is defined as CORE RESIDENTS not in the labour
-- force. That is a population definition, not a correction, and the paper must
-- state it: results describe people whose day is anchored in the core, either
-- because they work there or because they live there.
--
-- NOTE (documented limitation): core residents who work OUTSIDE the core are
-- in neither group. They are home-anchored in the core in the evening, so they
-- would belong here, but LODES pairs were filtered to work-block-in-core and
-- no longer carry them.

CREATE TABLE IF NOT EXISTS public.metro_home_blocks (
    metro       text     NOT NULL,
    geoid       char(15) NOT NULL,
    tract_geoid char(11) NOT NULL,
    pop20       integer  NOT NULL,      -- 2020 census block population = draw weight
    lat         double precision NOT NULL,
    lon         double precision NOT NULL,
    PRIMARY KEY (metro, geoid)
);

CREATE INDEX IF NOT EXISTS metro_home_blocks_tract_ix
    ON public.metro_home_blocks (metro, tract_geoid);

COMMENT ON TABLE public.metro_home_blocks IS
    'Populated census blocks inside each metro core. Two consumers: the '
    'non-worker home draw in db/build_population.py, and the routing endpoint '
    'universe (welfare_rs/routing_matrix.py), which must cover every node an '
    'agent home can snap to or those agents fall onto the slow lazy path.';

-- ── Reweighted PUMS: the tract-level joint distribution ──────────────────────
-- One row per (tract, PUMS household): how many households of that kind the
-- fitting thinks this tract holds. Produced by db/build_tract_weights.py with
-- IPU (Ye et al. 2009), which fits household and person controls together.
--
-- Stored rather than recomputed because it is the expensive, reusable artefact:
-- every replicate and every agent count draws from the same weights, and a
-- stored table can be verified against ACS independently of any population.

CREATE TABLE IF NOT EXISTS census.pums_tract_weights (
    tract_geoid char(11) NOT NULL,
    serialno    text     NOT NULL,
    weight      double precision NOT NULL,
    PRIMARY KEY (tract_geoid, serialno)
);

CREATE TABLE IF NOT EXISTS census.pums_tract_weight_meta (
    tract_geoid   char(11) PRIMARY KEY,
    puma          char(5)  NOT NULL,
    n_units       integer  NOT NULL,   -- PUMS records carrying weight
    iterations    integer  NOT NULL,
    max_rel_error double precision NOT NULL,
    converged     boolean  NOT NULL,
    built_at      timestamptz NOT NULL DEFAULT now()
);

COMMENT ON TABLE census.pums_tract_weight_meta IS
    'Per-tract IPU diagnostics. A tract with converged = false was fitted as '
    'well as the sample allows and still misses a control; the population '
    'build uses it but scripts/verify_acs_population.py reports the count.';

-- ── The built populations ────────────────────────────────────────────────────
-- What the engine reads. One row per agent, built offline, never sampled at
-- run time. `replicate` preserves what a simulation seed used to do: seed s
-- reads replicate s, so seeds still vary the population as well as behaviour.
--
-- ROW ORDER IS CONTRACTUAL, as it already is for personas: agent i is the row
-- with agent_id = i, and the paired No-RS counterfactual compares agent i
-- across conditions. A run of N agents reads agent_id < N, so a smaller run is
-- a prefix of a larger one and needs no rebuild.

CREATE TABLE IF NOT EXISTS public.metro_population (
    metro      text     NOT NULL,
    replicate  smallint NOT NULL,
    agent_id   integer  NOT NULL,

    is_worker  boolean  NOT NULL,

    -- Geography. Workers: both ends from one LODES pair. Non-workers: a core
    -- block weighted by population, and NULL work (the engine anchors their
    -- day at home).
    home_geoid char(15) NOT NULL,
    home_tract char(11) NOT NULL,
    home_lat   double precision NOT NULL,
    home_lon   double precision NOT NULL,
    work_geoid char(15),
    work_lat   double precision,
    work_lon   double precision,
    puma       char(5)  NOT NULL,

    -- The PUMS person this agent is a copy of.
    serialno   text     NOT NULL,
    sporder    smallint NOT NULL,
    age        smallint NOT NULL,
    sex        smallint,
    employed   boolean  NOT NULL,
    industry_seg smallint,                -- 1/2/3 = LODES SI01/02/03
    own_earnings double precision,         -- PUMS PERNP, 2023 dollars
    hh_income    double precision,         -- PUMS HINCP, 2023 dollars
    vehicles     smallint,
    disability   boolean,                  -- PUMS DIS: any disability
    -- PUMS DPHY: ambulatory difficulty specifically. `disability` is too broad
    -- to drive mobility_needs — it includes hearing, vision and cognitive
    -- difficulty, none of which changes how far someone can walk to a venue.
    ambulatory   boolean,
    hh_size      smallint,
    commute_mode smallint,                 -- PUMS JWTRNS, this person's own mode
    -- Two commute times, kept side by side so the match can be audited:
    -- what the drawn LODES pair implies for a car, and what this person
    -- actually reported (PUMS JWMNP). The draw matches them by band, so a
    -- long pair lands on someone who reports a long commute.
    commute_min_implied  double precision,
    commute_min_reported smallint,
    -- Share of the HOME TRACT's workers commuting by public transport
    -- (ACS B08301). A neighbourhood property, not a personal one, which is
    -- what `transit_access_level` has always meant: how well served this agent
    -- is, not how it happens to travel. Live once CAR_ONLY_MODE is removed.
    transit_share double precision,

    -- The survey respondent supplying psychometrics and attitudes.
    respondent_id text    NOT NULL,
    match_keys    text    NOT NULL,        -- which keys survived the match
    match_level   smallint NOT NULL,       -- 4 = all keys, 0 = tract-only fallback

    PRIMARY KEY (metro, replicate, agent_id)
);

COMMENT ON TABLE public.metro_population IS
    'Pre-built agent populations: LODES geography x reweighted PUMS person x '
    'matched survey respondent. Read by welfare_rs.datasource at run time; '
    'built by db/build_population.py.';

GRANT SELECT ON public.metro_home_blocks, public.metro_population TO welfare_app;

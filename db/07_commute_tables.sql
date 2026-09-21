-- Commute staging + the two serving tables. Run as welfare_owner:
--   psql -v ON_ERROR_STOP=1 -f db/07_commute_tables.sql
--
-- ONE filtered rowset, TWO levels of aggregation:
--
--   census.lodes_od            all OD rows for core-touching work counties
--        |  ST_Contains(geo.metro_cores.geom, census.blocks.pt)
--        v
--   census.od_core             rows whose WORK block is inside a core  <- staging
--        |                                                                  |
--        | GROUP BY home county                            keep each pair   |
--        v                                                                  v
--   public.metro_county_flows                             public.metro_od_pairs
--   (a few hundred rows;                                  (millions of rows;
--    sizes the road graph)                                 places the agents)
--
-- metro_county_flows is NOT a stopgap that metro_od_pairs later replaces. The
-- county list decides which counties get downloaded from Overpass, i.e. the
-- physical extent of the road graph -- and a home block in a county outside
-- that graph has no nearby node to snap to. So the county rollup must exist
-- BEFORE any home can be placed. Both are derived from od_core in one
-- transaction so they cannot drift.

-- ── Staging: which blocks lie inside which core ──────────────────────────────
-- Computed once and reused, because the SAME test is needed twice: on the WORK
-- block (does this job sit in the core?) and on the HOME block (does this
-- commuter already live in the core? -> core_share). Materialising it keeps
-- the expensive ST_Contains off both derivations.

CREATE TABLE IF NOT EXISTS census.core_blocks (
    metro   text     NOT NULL,
    geoid20 char(15) NOT NULL,
    PRIMARY KEY (metro, geoid20)
);

CREATE INDEX IF NOT EXISTS core_blocks_geoid_ix ON census.core_blocks (geoid20);

-- ── Staging: OD rows whose work block lands in a core ────────────────────────

CREATE TABLE IF NOT EXISTS census.od_core (
    metro     text     NOT NULL,
    w_geocode char(15) NOT NULL,
    h_geocode char(15) NOT NULL,
    jobs      integer  NOT NULL,      -- S000
    sa01 integer, sa02 integer, sa03 integer,   -- age  <=29 / 30-54 / 55+
    se01 integer, se02 integer, se03 integer,   -- earn <=1250 / 1251-3333 / >3333
    si01 integer, si02 integer, si03 integer,   -- industry: goods / trade-transp-util / other
    PRIMARY KEY (metro, w_geocode, h_geocode)
);

CREATE INDEX IF NOT EXISTS od_core_home_county_ix
    ON census.od_core (metro, left(h_geocode, 5));

-- ── Serving table 1: county flows ────────────────────────────────────────────
-- Schema is FIXED by the docstring in welfare_rs/datasource.py, which issues
--     SELECT county_fips, county_name, geocode, workers, core_share
--     FROM metro_county_flows WHERE metro = %s ORDER BY workers DESC
-- UNQUALIFIED, so it must live in `public` (the default search_path).
--
-- geocode is a Nominatim query string, because welfare_rs/metro.py feeds it
-- straight to geocode_polygon() to fetch that county's shell polygon. It is
-- built from the TIGER county + state names rather than hand-written, so it
-- cannot disagree with the FIPS it sits beside.
--
-- core_share was a hand-guessed constant when this table was provisional. It
-- is now measured: the share of that county's inbound workers whose HOME block
-- also falls inside the core polygon.

CREATE TABLE IF NOT EXISTS public.metro_county_flows (
    metro       text    NOT NULL,
    county_fips text    NOT NULL,
    county_name text    NOT NULL,
    geocode     text    NOT NULL,
    workers     integer NOT NULL,
    core_share  real    NOT NULL DEFAULT 0,
    PRIMARY KEY (metro, county_fips)
);

-- ── Serving table 2: block-level commute pairs ───────────────────────────────
-- The distribution agents are drawn from. A row is "this many jobs exist on
-- this home-block -> work-block pair", NOT an agent: a run draws num_agents
-- samples from it weighted by `jobs`, so agent count stays a run parameter.
--
-- Coordinates are denormalised (no join at run time) and are the Census
-- INTERNAL POINT of each block, which is guaranteed to lie inside the block --
-- unlike a centroid, which can fall outside a concave or water-wrapped one.
-- The simulation snaps them to graph nodes.
--
-- Age, earnings AND industry segments ride along. The ACS/PUMS population
-- build conditions on all three: a drawn pair fixes the job's age band,
-- earnings band and industry sector, and the matching PUMS person must agree
-- with them. Industry was originally left out because nothing read it; the
-- population build reads it (db/11_industry_columns.sql added it in place).

-- h_county is stored rather than sliced from h_geoid at run time because the
-- sampler MUST restrict to counties present in the road graph: the graph only
-- covers the counties that passed the 90% coverage rule, and a home outside it
-- would snap to whatever node sits on the graph boundary.

CREATE TABLE IF NOT EXISTS public.metro_od_pairs (
    metro    text     NOT NULL,
    h_geoid  char(15) NOT NULL,
    w_geoid  char(15) NOT NULL,
    h_county char(5)  NOT NULL,
    jobs     integer  NOT NULL,
    sa01 integer, sa02 integer, sa03 integer,
    se01 integer, se02 integer, se03 integer,
    si01 integer, si02 integer, si03 integer,
    h_lat double precision NOT NULL,
    h_lon double precision NOT NULL,
    w_lat double precision NOT NULL,
    w_lon double precision NOT NULL,
    PRIMARY KEY (metro, h_geoid, w_geoid)
);

GRANT SELECT ON ALL TABLES IN SCHEMA census, public TO welfare_app;

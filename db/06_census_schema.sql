-- Census staging geography + LODES origin-destination. Run as welfare_owner:
--   psql -v ON_ERROR_STOP=1 -f db/06_census_schema.sql
--
-- `census` is the STAGING tier for the commute pipeline, the same way `poi`
-- stages pois_us for public.metro_pois. Nothing here is read at run time; it
-- exists so the two serving tables (public.metro_county_flows and
-- public.metro_od_pairs) can be re-derived cheaply while the model iterates.
--
-- Division of labour vs the `geo` schema — they do NOT overlap:
--   * geo.metro_cores  = 8 city polygons from Nominatim. A MODEL CONCEPT:
--     "the principal city", the only place POIs and jobs may exist.
--   * census.blocks    = millions of Census block polygons. A DICTIONARY:
--     it resolves a LODES 15-digit GEOID to a coordinate.
-- They meet at exactly one predicate:
--     ST_Contains(geo.metro_cores.geom, census.blocks.pt)
-- which is what turns "block 110010062021003" into "a job inside the DC core".
--
-- Census GEOIDs nest by string prefix, so this one block table also joins to
-- every coarser level without storing them:
--   block 15 = state 2 | county 3 | tract 6 | block 4
--   block group = left(geoid20, 12)   tract = left(geoid20, 11)
--   county      = left(geoid20, 5)    state = left(geoid20, 2)
-- That is what a later ACS build (block-group marginals) will join on.

CREATE SCHEMA IF NOT EXISTS census AUTHORIZATION welfare_owner;

GRANT USAGE ON SCHEMA census TO welfare_app;
-- Applies to tables created AFTER this statement, hence before the CREATEs.
ALTER DEFAULT PRIVILEGES FOR ROLE welfare_owner IN SCHEMA census
    GRANT SELECT ON TABLES TO welfare_app;

-- ── TIGER states / counties ──────────────────────────────────────────────────
-- Small national files. counties supplies both the county NAME for
-- metro_county_flows and the state-intersection test that decides which
-- states' block files we need to download at all.

CREATE TABLE IF NOT EXISTS census.states (
    statefp  char(2) PRIMARY KEY,
    usps     char(2) NOT NULL,       -- 'MD'
    name     text    NOT NULL,       -- 'Maryland'
    geom     geometry(MultiPolygon, 4326) NOT NULL
);

CREATE TABLE IF NOT EXISTS census.counties (
    geoid    char(5) PRIMARY KEY,    -- statefp || countyfp
    statefp  char(2) NOT NULL,
    name     text    NOT NULL,       -- 'Montgomery'
    namelsad text    NOT NULL,       -- 'Montgomery County'
    aland    bigint,
    awater   bigint,
    geom     geometry(MultiPolygon, 4326) NOT NULL
);

CREATE INDEX IF NOT EXISTS counties_geom_gix ON census.counties USING gist (geom);
CREATE INDEX IF NOT EXISTS counties_statefp_ix ON census.counties (statefp);

-- ── TIGER 2020 blocks (tabblock20) ───────────────────────────────────────────
-- LODES8 keys on 2020 blocks, so this MUST be the 2020 vintage. (The
-- postgis_tiger_geocoder extension ships a `tabblock` table, but it is
-- 2010-vintage; 2010 and 2020 GEOIDs are both 15 digits and would join without
-- erroring, silently against redrawn geography. Hence our own table.)
--
-- `pt` is the Census-published INTERNAL POINT (INTPTLAT20/INTPTLON20), not a
-- computed centroid: it is guaranteed to fall inside the block, whereas a
-- centroid of a concave or water-wrapped block can fall outside it.

CREATE TABLE IF NOT EXISTS census.blocks (
    geoid20    char(15) PRIMARY KEY,
    statefp    char(2)  NOT NULL,
    county_fips char(5) NOT NULL,    -- statefp || countyfp, the flows join key
    aland      bigint,
    awater     bigint,
    pop20      integer,
    housing20  integer,
    pt         geometry(Point, 4326) NOT NULL,
    geom       geometry(MultiPolygon, 4326)
);

-- pt carries the core-membership test; geom is kept for later ACS/areal work.
CREATE INDEX IF NOT EXISTS blocks_pt_gix ON census.blocks USING gist (pt);
CREATE INDEX IF NOT EXISTS blocks_county_ix ON census.blocks (county_fips);

-- ── LODES8 origin-destination ────────────────────────────────────────────────
-- One row per (work block, home block) pair with job counts, as published.
-- Vintage is pinned per load and recorded so a re-derive can never mix years.
--
--   part='main' : worker lives in the same state as the job
--   part='aux'  : worker lives OUT of state  <- ESSENTIAL, not optional.
--     Verified on dc_od_JT01_2023: main = 203,088 jobs, aux = 422,114.
--     Two thirds of DC's workforce is in aux; dropping it would delete
--     Maryland and Virginia from the model.
--
-- Segment columns are kept in staging even where the model has no consumer yet
-- (industry), because re-downloading to recover them costs more than the disk.
--   sa01/02/03  age       <=29 / 30-54 / 55+
--   se01/02/03  earnings  <=$1250/mo / $1251-3333 / >$3333
--   si01/02/03  industry  goods / trade-transport-utilities / other services

CREATE TABLE IF NOT EXISTS census.lodes_od (
    w_geocode char(15) NOT NULL,     -- work block
    h_geocode char(15) NOT NULL,     -- home block
    part      char(4)  NOT NULL,     -- 'main' | 'aux'
    st        char(2)  NOT NULL,     -- state whose file this row came from
    year      smallint NOT NULL,
    jt        char(4)  NOT NULL,     -- job type, e.g. 'JT01' (primary jobs)
    s000 integer NOT NULL,
    sa01 integer, sa02 integer, sa03 integer,
    se01 integer, se02 integer, se03 integer,
    si01 integer, si02 integer, si03 integer
);

-- Indexes are created by db/08_index_lodes_od.sql AFTER the bulk COPY.

GRANT SELECT ON ALL TABLES IN SCHEMA census TO welfare_app;

-- Shared spatial reference geography. Run as welfare_owner:
--   psql -v ON_ERROR_STOP=1 -f db/04_geo_schema.sql
--
-- `geo` holds boundary polygons that MORE THAN ONE pipeline needs:
--   * clipping poi.pois_us -> public.metro_pois       (POI step)
--   * testing which LODES work blocks fall in a core  (flows step)
--   * the core layer of the two-layer metro network   (welfare_rs/metro.py)
-- Those three MUST agree on the polygon, so it is stored once and queried,
-- never re-derived per pipeline.
--
-- TIGER county/block tables will join this schema later; they are not created
-- here because the county list is an OUTPUT of the LODES flow computation.

CREATE SCHEMA IF NOT EXISTS geo AUTHORIZATION welfare_owner;

GRANT USAGE ON SCHEMA geo TO welfare_app;
-- Applies to tables created AFTER this statement, hence before the CREATEs.
ALTER DEFAULT PRIVILEGES FOR ROLE welfare_owner IN SCHEMA geo
    GRANT SELECT ON TABLES TO welfare_app;

-- One row per Nominatim query in params.METRO_PARAMS[metro]["core_places"].
-- Kept separate from the union so a suspect boundary can be inspected and
-- re-fetched on its own (e.g. sf_bay = San Francisco + Oakland).
CREATE TABLE IF NOT EXISTS geo.metro_core_parts (
    metro        text NOT NULL,   -- key from params.METRO_PARAMS["metros"]
    place_query  text NOT NULL,   -- the exact geocode string, verbatim
    display_name text,            -- what Nominatim resolved it to
    fetched_at   timestamptz NOT NULL DEFAULT now(),
    geom         geometry(MultiPolygon, 4326) NOT NULL,
    PRIMARY KEY (metro, place_query)
);

-- One row per metro: the union of its parts. THE table other pipelines query.
CREATE TABLE IF NOT EXISTS geo.metro_cores (
    metro text PRIMARY KEY,
    label text NOT NULL,
    parts integer NOT NULL,       -- how many core_places were unioned
    geom  geometry(MultiPolygon, 4326) NOT NULL
);

CREATE INDEX IF NOT EXISTS metro_cores_geom_gix ON geo.metro_cores USING gist (geom);

GRANT SELECT ON ALL TABLES IN SCHEMA geo TO welfare_app;

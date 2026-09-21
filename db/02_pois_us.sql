-- Raw all-US POI reference table (mirrors data/pois.parquet). No gist index yet
-- (built after the bulk load in 03). Run as welfare_owner over TCP:
--   psql -v ON_ERROR_STOP=1 -f db/02_pois_us.sql
-- with PGHOST/PGDATABASE/PGUSER/PGPASSWORD exported (see the flow in chat).

CREATE SCHEMA IF NOT EXISTS poi;

CREATE TABLE IF NOT EXISTS poi.pois_us (
    id                   bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    placekey             text,   -- raw; '|'-separated on shared_polygon rows
    parent_placekey      text,
    location_name        text,   -- raw; '|'-separated on shared_polygon rows
    naics_code           text,   -- raw; '|'-separated on shared_polygon rows
    top_category         text,
    sub_category         text,
    top_category_2digit  text,
    top_category_final   text,
    state                text,
    city                 text,
    postal_code          text,
    street_address       text,
    latitude             double precision,
    longitude            double precision,
    geom  geometry(Point,4326)
          GENERATED ALWAYS AS (ST_SetSRID(ST_MakePoint(longitude, latitude), 4326)) STORED,
    polygon_class        text,
    enclosed             boolean,
    includes_parking_lot boolean,
    geometry_type        text,
    wkt_area_sq_meters   double precision
    -- POLYGON_WKT (building footprints) intentionally dropped: heavy, unused by the sim.
);

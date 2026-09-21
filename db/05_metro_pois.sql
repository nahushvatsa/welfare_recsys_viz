-- The app's POI serving table. Run as welfare_owner:
--   psql -v ON_ERROR_STOP=1 -f db/05_metro_pois.sql
--
-- Schema is fixed by the docstring in welfare_rs/datasource.py and must not
-- drift: PostgresDataSource.poi_rows() issues
--     SELECT place_id, name, category, latitude, longitude
--     FROM metro_pois WHERE metro = %s
-- UNQUALIFIED, so this table has to live in `public` (the default search_path).
--
-- Populated by db/derive_metro_pois.py, which clips poi.pois_us against
-- geo.metro_cores. Rows are NOT capped here: params.POI_PARAMS["max_per_category"]
-- is applied at run time in simulation.py with a fixed seed, so the POI set stays
-- identical across treatments and seeds. Capping here would break that.
--
-- No extra index: PRIMARY KEY (metro, place_id) is a btree whose leading column
-- is `metro`, which is the only filter the app ever applies.

CREATE TABLE IF NOT EXISTS public.metro_pois (
    metro     text NOT NULL,             -- key from params.METRO_PARAMS["metros"]
    place_id  text NOT NULL,             -- Placekey (single, already un-packed)
    name      text NOT NULL,
    category  text NOT NULL,             -- leisure key from params.CATEGORY_KEYWORDS
    latitude  double precision NOT NULL,
    longitude double precision NOT NULL,
    PRIMARY KEY (metro, place_id)
);

-- Build indexes AFTER the bulk load (much faster than maintaining them during COPY).
-- Run as welfare_owner over TCP:  psql -v ON_ERROR_STOP=1 -f db/03_index_pois_us.sql

SET maintenance_work_mem = '2GB';   -- session-only; speeds the gist build

CREATE INDEX IF NOT EXISTS pois_us_geom_gix  ON poi.pois_us USING gist (geom);
CREATE INDEX IF NOT EXISTS pois_us_state_ix  ON poi.pois_us (state);

ANALYZE poi.pois_us;

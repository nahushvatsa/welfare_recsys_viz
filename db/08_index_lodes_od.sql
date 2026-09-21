-- Index census.lodes_od AFTER the bulk COPY (cheaper than maintaining it
-- during the load). Run as welfare_owner:
--   psql -v ON_ERROR_STOP=1 -f db/08_index_lodes_od.sql
--
-- derive_metro_flows.py's `od-core` phase joins lodes_od.w_geocode against
-- census.core_blocks once per metro. Without this index that is eight full
-- scans of a ~10M-row table; with it each metro touches only its own work
-- blocks.

SET maintenance_work_mem = '2GB';   -- session-only; speeds the build

CREATE INDEX IF NOT EXISTS lodes_od_w_geocode_ix ON census.lodes_od (w_geocode);

-- Home-side lookups (coverage checks, home-state discovery) scan by state
-- prefix rather than by the whole GEOID.
CREATE INDEX IF NOT EXISTS lodes_od_h_state_ix ON census.lodes_od (left(h_geocode, 2));

ANALYZE census.lodes_od;

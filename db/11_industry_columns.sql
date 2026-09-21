-- Carry LODES industry segments (SI01-03) into the derived commute tables.
-- Run as welfare_owner:  psql -v ON_ERROR_STOP=1 -f db/11_industry_columns.sql
--
-- census.lodes_od has held si01-03 since the original load; only the two
-- derived tables dropped them, on the reasoning that nothing in the model read
-- industry. The ACS/PUMS population build does: a LODES pair now supplies the
-- age, earnings AND industry segment of the job, and PUMS carries industry per
-- person, so matching on it sharpens the draw at no data cost.
--
-- Nullable and with no default, so the ALTER is a catalog-only change on the
-- 7.9M-row serving table. db/derive_metro_flows.py fills them on the next
-- od-core + pairs re-derive; until then they are NULL and every consumer
-- treats a NULL segment as "unconditioned", exactly as it treats the absence
-- of segment counts today.

ALTER TABLE census.od_core
    ADD COLUMN IF NOT EXISTS si01 integer,
    ADD COLUMN IF NOT EXISTS si02 integer,
    ADD COLUMN IF NOT EXISTS si03 integer;

ALTER TABLE public.metro_od_pairs
    ADD COLUMN IF NOT EXISTS si01 integer,
    ADD COLUMN IF NOT EXISTS si02 integer,
    ADD COLUMN IF NOT EXISTS si03 integer;

COMMENT ON COLUMN public.metro_od_pairs.si01 IS
    'LODES SI01 jobs: goods producing';
COMMENT ON COLUMN public.metro_od_pairs.si02 IS
    'LODES SI02 jobs: trade, transportation and utilities';
COMMENT ON COLUMN public.metro_od_pairs.si03 IS
    'LODES SI03 jobs: all other services';

GRANT SELECT ON census.od_core, public.metro_od_pairs TO welfare_app;

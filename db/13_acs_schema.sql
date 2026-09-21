-- ACS 5-year detailed tables, long format, for the population reweighting.
-- Run as welfare_owner:  psql -v ON_ERROR_STOP=1 -f db/13_acs_schema.sql
--
-- These are the small-area CONTROL TOTALS the PUMS sample is fitted to. PUMS
-- knows people jointly but only to a PUMA (>=100k people); ACS knows a tract
-- or block group precisely but only one margin at a time. Fitting one to the
-- other is what gives an agent both: a real person's internally consistent
-- attributes, at the local rates of the neighbourhood it actually lives in.
--
-- WHY LONG AND NOT WIDE. The eight tables have between 7 and 49 variables each
-- and publish at different geographic levels; a wide table would be mostly
-- NULL and would need a migration every time a control is added or dropped.
-- The reweighting reads one metro's controls in a single query and pivots in
-- numpy, so the long form costs nothing at use time.
--
-- FIT AT TRACT, NOT BLOCK GROUP. Measured on Manhattan's 1,292 block groups:
-- the average block group holds ~600 households with a margin of error of
-- +/-33% on that total, and a single income bucket inside it carries a margin
-- of error of 134% OF THE ESTIMATE. Fitting weights to numbers that noisy
-- fits the noise. Tracts are ~4x larger and hold up. Block-group rows are
-- loaded anyway (they cost little and two tables below are BG-published), but
-- the reweighting uses sumlevel 140.

CREATE TABLE IF NOT EXISTS census.acs_estimates (
    geoid    text     NOT NULL,   -- 11 digits (tract) or 12 (block group)
    sumlevel smallint NOT NULL,   -- 140 tract, 150 block group, 50 county
    table_id text     NOT NULL,   -- e.g. 'b19001'
    varnum   smallint NOT NULL,   -- the NNN of B19001_E0NN
    estimate double precision,    -- NULL when the cell is suppressed
    moe      double precision,    -- margin of error, 90% confidence
    PRIMARY KEY (geoid, table_id, varnum)
);

CREATE INDEX IF NOT EXISTS acs_estimates_table_ix
    ON census.acs_estimates (table_id, sumlevel);

COMMENT ON TABLE census.acs_estimates IS
    'ACS 2019-2023 5-year detailed tables for the 7 metro states, long format. '
    'Control totals for the PUMS reweighting in db/build_population.py.';

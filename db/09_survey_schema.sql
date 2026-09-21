-- Prolific stated-preference survey package -> the `survey` schema.
-- Run as welfare_owner:
--   psql -v ON_ERROR_STOP=1 -f db/09_survey_schema.sql
--
-- Source: data_handoff/, the August 2026 package from Ekin Ugurel covering the
-- April 2026 survey (IRB-FY2026-11354, NYU Tandon). N=473 after consent,
-- completion and four attention checks.
--
-- ── HUMAN-SUBJECTS DATA. READ BEFORE CHANGING A GRANT. ──────────────────────
-- survey.respondents is respondent-level human-subjects data. The handoff
-- terms (data_handoff/README.md S9) require institutional storage and prohibit
-- redistribution. The public-facing API runs as welfare_app.
--
-- So, unlike `census` and `poi`, this schema does NOT set
--     ALTER DEFAULT PRIVILEGES ... GRANT SELECT ON TABLES TO welfare_app;
-- welfare_app gets USAGE on the schema and SELECT on exactly ONE relation:
-- the survey.personas VIEW (db/10_survey_personas.sql), which exposes the 53
-- columns the ABM consumes and nothing else. A Postgres view executes with its
-- OWNER's privileges, so welfare_app reads the view without ever holding
-- SELECT on the 136-column respondent file underneath it. That property is the
-- access control; do not "fix" it by granting the base table.
--
-- ── Division of labour ──────────────────────────────────────────────────────
-- Everything here is the shipped package, loaded verbatim and checksummed
-- against manifest.json. Nothing in this schema is derived by us EXCEPT the
-- personas view, which lives in its own file. Keeping the package immutable is
-- what lets db/load_survey.py re-verify at any time that what the DB holds is
-- what Ekin shipped.

CREATE SCHEMA IF NOT EXISTS survey AUTHORIZATION welfare_owner;

GRANT USAGE ON SCHEMA survey TO welfare_app;
REVOKE ALL ON ALL TABLES IN SCHEMA survey FROM welfare_app;


-- ── Provenance ───────────────────────────────────────────────────────────────
-- manifest.json. sha256 is what makes "the DB matches the package" checkable
-- years from now, after the working copy has been moved or re-exported.

CREATE TABLE IF NOT EXISTS survey.package_files (
    name        text PRIMARY KEY,
    bytes       bigint NOT NULL,
    sha256      char(64) NOT NULL,
    description text,
    verified_at timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT package_files_sha256_hex CHECK (sha256 ~ '^[0-9a-f]{64}$')
);

COMMENT ON TABLE survey.package_files IS
    'manifest.json: every shipped file with its size and SHA-256. Re-verified '
    'on every run of db/load_survey.py; a mismatch aborts the load.';


-- ── The data dictionary ──────────────────────────────────────────────────────
-- One row per column of the respondent file, including verbatim item wording.
-- This is why item text never has to be hardcoded anywhere else.

CREATE TABLE IF NOT EXISTS survey.data_dictionary (
    variable           text PRIMARY KEY,
    block              text,
    source_question    text,
    data_type          text,
    valid_values       text,
    n_missing          integer NOT NULL,
    missing_convention text,
    latent_construct   text,   -- '' when the item joins no composite
    reverse_coded      text,
    description        text
);

COMMENT ON COLUMN survey.data_dictionary.n_missing IS
    'Declared missing count. db/load_survey.py asserts this equals the observed '
    'count in survey.respondents for all 136 columns.';


-- ── Composite scoring map ────────────────────────────────────────────────────
-- construct_items.json. Every composite is an UNWEIGHTED MEAN of these items,
-- with reverse-coded items already reversed in their _R columns (README S6).
-- Storing it makes each composite re-derivable in SQL rather than on trust.

CREATE TABLE IF NOT EXISTS survey.construct_items (
    construct text NOT NULL,
    item      text NOT NULL,
    item_ord  smallint NOT NULL,   -- position in the shipped JSON array
    PRIMARY KEY (construct, item),
    FOREIGN KEY (item) REFERENCES survey.data_dictionary (variable)
);


-- ── Calibration targets ──────────────────────────────────────────────────────
-- The joint distribution a synthetic population is supposed to reproduce
-- (README S4). We do NOT synthesise a population -- the ABM uses the observed
-- respondents themselves -- so these are held as the yardstick that
-- scripts/verify_survey_population.py check 6 measures the built population
-- against, not as an input.

CREATE TABLE IF NOT EXISTS survey.marginals (
    variable   text NOT NULL,
    kind       text NOT NULL,   -- 'categorical' | 'numeric'
    level      text,            -- category label, or the moment/quantile name
    statistic  text,
    value      double precision,
    proportion double precision
);

CREATE INDEX IF NOT EXISTS marginals_variable_ix ON survey.marginals (variable);

-- Long form, not a 33x33 matrix: the pairwise N belongs beside its r, and
-- 'give me r for these two variables' is the only query anyone runs.
CREATE TABLE IF NOT EXISTS survey.correlations (
    var_a     text NOT NULL,
    var_b     text NOT NULL,
    pearson_r double precision NOT NULL,
    pairwise_n integer NOT NULL,
    PRIMARY KEY (var_a, var_b),
    CONSTRAINT correlations_r_range CHECK (pearson_r BETWEEN -1 AND 1)
);

COMMENT ON TABLE survey.correlations IS
    'calibration_correlations.csv x _n.csv, joined. Pearson over the 463 '
    'listwise-complete cases; PSD (min eigenvalue 0.0318).';

-- calibration_correlations_meta.json, verbatim. Single row.
CREATE TABLE IF NOT EXISTS survey.calibration_meta (
    only_row boolean PRIMARY KEY DEFAULT true CHECK (only_row),
    meta     jsonb NOT NULL
);


-- survey.respondents is created by db/load_survey.py, which generates its 136
-- typed columns from survey.data_dictionary and writes the emitted DDL to
-- db/09b_survey_respondents.generated.sql for review. It is generated rather
-- than hand-written because 136 hand-typed columns is 136 chances to mistype a
-- scale.

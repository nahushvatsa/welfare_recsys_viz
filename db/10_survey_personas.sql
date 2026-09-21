-- survey.personas — the ABM-facing projection of the survey. Run as owner:
--   psql -v ON_ERROR_STOP=1 -f db/10_survey_personas.sql
--
-- One row per respondent, in the 53-column layout
-- welfare_rs.simulation.Simulation._load_personas consumes. This is the SQL
-- twin of data/build_survey_personas.py, which remains the offline path for
-- machines with no DB; scripts/verify_survey_personas_view.py asserts the two
-- agree cell for cell, so a change to either that is not made to the other
-- fails loudly.
--
-- WHY A VIEW, NOT A TABLE. The population is the survey sample itself, not a
-- synthesised draw, so there is nothing to materialise: 473 rows recomputed on
-- demand cost nothing, and a view cannot drift from survey.respondents the way
-- a copy can.
--
-- ACCESS CONTROL. This is the ONE relation in the schema welfare_app may read.
-- A Postgres view runs with its owner's privileges, so welfare_app selects
-- these 53 columns without holding SELECT on the 136-column respondent file
-- underneath. Do not grant the base table to make something else work.
--
-- Every column here is survey-derived. Car access, transit access, mobility
-- needs, walking tolerance, primary leisure interest, preferred time window,
-- travel-party composition, environmental attitude and risk salience have NO
-- counterpart anywhere in the instrument (checked against all 136 columns) and
-- are deliberately absent: the simulation holds them fixed for the whole
-- population via params.PERSONA_MAPPING["fixed_attributes"]. See
-- docs/survey-coverage.md S2 for why constants beat draws from a D-optimal
-- design that was balanced by construction rather than representative.

DROP VIEW IF EXISTS survey.personas;

CREATE VIEW survey.personas AS
WITH imputed AS (
    -- Monthly leisure spend drives both Budget and budget_tightness. Five
    -- respondents chose "I'm not sure" and are missing in the shipped file
    -- (README S7); they take the sample median so neither field silently
    -- defaults. percentile_cont interpolates at 0.5, which for the 468
    -- non-missing values equals the mean of the two central order statistics —
    -- the same definition data/build_survey_personas.py uses. It resolves to
    -- exactly 3.0 here, so the imputed agents get budget_tightness 0.5 and
    -- Budget 'medium'.
    SELECT percentile_cont(0.5) WITHIN GROUP (
               ORDER BY "LeisureSpend_num"::double precision
           ) AS spend_median
    FROM survey.respondents
    WHERE "LeisureSpend_num" IS NOT NULL
),
base AS (
    SELECT r.*,
           COALESCE(r."LeisureSpend_num"::double precision, i.spend_median) AS spend
    FROM survey.respondents r CROSS JOIN imputed i
)
SELECT
    -- ── Identity ────────────────────────────────────────────────────────────
    b.respondent_id                        AS "PersonaID",

    -- ── Demographics the agent builder reads ────────────────────────────────
    -- Band LABELS, not codes: params.PERSONA_MAPPING carries the survey's own
    -- band strings and converts them to years and dollars. Age_num and
    -- Income_num are ordinal ranks and must NOT be used for that (README S5).
    b."Age"                                AS "Age",
    b."HouseholdIncome"                    AS "Income",
    b."Sex"                                AS "Sex",

    -- ── Derived: budget ─────────────────────────────────────────────────────
    -- [0,1] proportion, 1 = tightest (< $50/month). The ::double precision on
    -- spend is load-bearing: LeisureSpend_num is smallint, and smallint
    -- arithmetic in Postgres is integer division, which would flatten this to
    -- {1.0, 0.75, 0.5} and quietly halve its spread. It is cast in `base`.
    round((1.0 - (b.spend - 1.0) / 4.0)::numeric, 6)::double precision
                                           AS budget_tightness,
    CASE WHEN b.spend <= 2 THEN 'low'
         WHEN b.spend <= 3 THEN 'medium'
         ELSE 'high' END                   AS "Budget",

    -- TopRated sets the base level of practice conformity, which the survey's
    -- popularity-seeking item then modulates through trend susceptibility.
    -- Deriving it from that same item keeps base and shift pointing the same
    -- way; drawing it independently would inject noise uncorrelated with the
    -- respondent's own stated preference for well-known destinations.
    CASE WHEN b."s5_general_a" >= 5 THEN 'yes' ELSE 'no' END
                                           AS "TopRated",

    -- ── Psychometric composites (unweighted means of 7-point items) ─────────
    b."Openness"                           AS openness,
    b."Conscientiousness"                  AS conscientiousness,
    b."Extraversion"                       AS extraversion,
    b."Agreeableness"                      AS agreeableness,
    b."Neuroticism"                        AS neuroticism,
    b."Maximization"                       AS maximization,
    b."Recommendation_Trust"               AS trust_platforms,
    b."Algorithmic_Awareness"              AS algorithmic_awareness,
    -- NOT the Autonomy_Control composite. That composite is positively loaded
    -- on liking personalisation (+0.46 with Recommendation_Trust, +0.43 with
    -- following platform recommendations), while this slot enters the
    -- acceptance probability with a NEGATIVE coefficient — wiring the
    -- composite here would invert beta_O. s6_autonomy_control_a ("I prefer to
    -- make my own leisure choices rather than follow platform suggestions") is
    -- the item that measures the construct the ABM needs (-0.52 with trust)
    -- and is deliberately excluded from every composite in the package.
    b."s6_autonomy_control_a"              AS autonomy_preference,

    -- ── Survey items, on their native response scales ───────────────────────
    -- Written raw (7-point as 1..7, not pre-normalised). The consumer
    -- normalises against welfare_rs.agent.SURVEY_ITEM_SCALES. Seven further
    -- item slots have no counterpart in this instrument and keep the agent
    -- default of 0.5 — see docs/survey-coverage.md S3 for what each damps.
    b."leisure_freq_num"                   AS local_leisure_frequency,
    b."overnight_num"                      AS overnight_trip_frequency,
    b."spur_num"                           AS spontaneity_share,
    b."CityFam_num"                        AS city_familiarity,
    b."s2_Q16"                             AS follow_through_friend,
    b."s2_Q17"                             AS follow_through_platform,
    b."s2_Q18"                             AS follow_through_ai,
    b."s4_search_c"                        AS cross_platform_search,
    b."s4_search_e"                        AS price_filter_tendency,
    b."s4_group_coord_mean"                AS group_coordination_preference,
    b."s5_general_a"                       AS popularity_herding,
    b."s3_attitudes_d"                     AS unexpected_discovery,
    b."s4_social_media_b"                  AS social_posting,
    b."s5_search_c"                        AS multi_platform_parallel,
    b."s6_trust_reliance_c"                AS ai_itinerary_comfort,
    b."s6_trust_reliance_d"                AS explanation_needed,
    b."s6_algo_knowledge_correct"          AS objective_algorithmic_literacy,

    -- ── Carried: recorded on Agent.characteristics, no behavioural effect ───
    -- Employment is the exception and is NOT inert: it decides whether an
    -- agent holds a job, which sets both the day structure (work-anchored vs
    -- home-anchored) and the value-of-time share. 161 of 473 respondents (34%)
    -- are retired, unemployed, out of the labour force or studying.
    b."Education"                          AS "Education",
    b."Education_num"                      AS "Education_num",
    b."Employment"                         AS "Employment",
    b."RaceEthnicity"                      AS "RaceEthnicity",
    b."HomeLanguage"                       AS "HomeLanguage",
    b."CensusRegion"                       AS "CensusRegion",
    b."CensusDivision"                     AS "CensusDivision",
    b."ResidenceLength"                    AS "ResidenceLength",
    b."ResidenceLength_num"                AS "ResidenceLength_num",
    b."LeisureSpend"                       AS "LeisureSpend",
    -- ::double precision is NOT redundant. LeisureSpend_num is a 1-5 ordinal
    -- and survey.respondents types it smallint, which is what it means. But
    -- the SHIPPED csv carries it float-formatted ("3.0"), because pandas
    -- upcast the column when five respondents answered "I'm not sure" -- and
    -- this value is copied verbatim, as a string, onto Agent.characteristics.
    -- Rendering "3" here instead of "3.0" would make an agent built from the
    -- DB carry a different characteristics value than the same agent built
    -- from the CSV: inert for behaviour, but it would break the
    -- interchangeability the whole two-path design rests on, and would silently
    -- split a group-by on that key. The cast reproduces the file's
    -- representation; it does not restore the file's typing mistake upstream.
    b."LeisureSpend_num"::double precision  AS "LeisureSpend_num",
    b."CityFam_num"                        AS "CityFam_num",
    b."Age_num"                            AS "Age_num",
    b."Income_num"                         AS "Income_num",
    b."Sex_num"                            AS "Sex_num",
    b."scenario"                           AS "scenario",
    -- Retained for analysis only; deliberately not wired to
    -- autonomy_preference, for the reason given above.
    b."Autonomy_Control"                   AS "Autonomy_Control",
    b."Platform_Comfort"                   AS "Platform_Comfort",

    -- ── Geography: none ─────────────────────────────────────────────────────
    -- The survey is a national sample with home ZIP removed for
    -- de-identification (README S9: 461 of 473 ZIPs were unique), so it
    -- carries no usable geography. Homes and workplaces come from the metro's
    -- LODES commute pairs; on an unlayered single-city network the loader
    -- falls back to a random network node. NULL here renders as the empty
    -- string the CSV path writes.
    NULL::double precision                 AS start_latitude,
    NULL::double precision                 AS start_longitude
FROM base b
ORDER BY b.respondent_id;

COMMENT ON VIEW survey.personas IS
    'ABM-facing projection of survey.respondents: one agent profile per '
    'respondent, 53 columns, matching data/build_survey_personas.py. The only '
    'relation in this schema readable by welfare_app.';

GRANT SELECT ON survey.personas TO welfare_app;

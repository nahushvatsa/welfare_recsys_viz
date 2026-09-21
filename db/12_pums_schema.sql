-- ACS PUMS (Public Use Microdata Sample) + the tract -> PUMA crosswalk.
-- Run as welfare_owner:  psql -v ON_ERROR_STOP=1 -f db/12_pums_schema.sql
--
-- WHY THIS EXISTS. LODES places an agent (block-level home and workplace) but
-- describes almost nobody: three age bands, three earnings bands, three
-- industry sectors, workers only. PUMS is the opposite — one real person's
-- age, own earnings, household income, vehicles, disability and household
-- composition, all TOGETHER, which is the only public source of the joint
-- distribution the persona needs. Its geography is coarse (a PUMA is >=100k
-- people, ~34 tracts), so the two are used for what each is good at:
-- LODES for where, PUMS for who, joined through census.tract_puma.
--
-- Vintage: ACS 2019-2023 5-year, matching LODES8 2023. Both sit on 2020
-- census geography, so no geography crosswalk is needed between them.
--
-- COLUMNS ARE A CURATED SUBSET. The source files carry ~290 (person) and
-- ~240 (household) columns, most of which are housing-quality, insurance and
-- allocation flags the ABM has no use for, plus 80 replicate weights each.
-- Kept here: everything the agent model reads, everything the reweighting
-- fits to, and the demographics carried for reporting. The raw .zip files stay
-- on disk under data/census/pums with SHA-256s in manifest_acs_pums.json, so
-- adding a column later is a re-run, not a re-download.

CREATE SCHEMA IF NOT EXISTS census;

-- ── Households ───────────────────────────────────────────────────────────────
-- One row per housing unit OR group-quarters record.
--
-- THREE RECORD KINDS, and the difference decides what may be fitted to what:
--   typehugq = 1  housing unit.     wgtp > 0, hincp present when occupied.
--   typehugq = 2  institutional GQ (prisons, nursing homes).   wgtp = 0.
--   typehugq = 3  non-institutional GQ (dorms, shelters).      wgtp = 0.
-- GQ records carry NO household weight and NO household income — but their
-- PERSONS carry full person weights. Verified on DC: 3,231 GQ persons, all
-- with pwgtp > 0, against 533 + 2,698 GQ household records all at wgtp = 0.
-- So household controls (income, vehicles, household type) may only be fitted
-- over occupied housing units, while person controls (age, sex, employment,
-- disability) must include GQ persons or every tract with a dormitory or a
-- nursing home is fitted wrong.
--
-- A vacant housing unit is typehugq = 1 with np = 0 and no income; ACS
-- household tables count OCCUPIED units, so `occupied` is the flag to fit on.

CREATE TABLE IF NOT EXISTS census.pums_households (
    serialno   text     NOT NULL,
    state      char(2)  NOT NULL,
    puma       char(5)  NOT NULL,
    wgtp       integer  NOT NULL,          -- household weight (0 for GQ)
    np         smallint NOT NULL,          -- persons in unit (0 = vacant)
    typehugq   smallint NOT NULL,

    -- Income adjustment. Five survey years share one file, so dollar amounts
    -- are in the dollars of the year they were reported in. Multiplying by
    -- adjinc / 1e6 puts them all in 2023 dollars; adjhsg does the same for
    -- housing costs. Skipping this silently mixes 2019 and 2023 dollars.
    adjinc     integer  NOT NULL,
    adjhsg     integer,

    hincp      integer,                    -- household income, as reported
    fincp      integer,                    -- family income, as reported
    hincp_adj  double precision GENERATED ALWAYS AS
                   (hincp::double precision * adjinc / 1000000.0) STORED,

    veh        smallint,   -- vehicles available (0-6); NULL for GQ and vacant
    ten        smallint,   -- tenure (own with mortgage / free and clear / rent / no rent)
    bld        smallint,   -- units in structure
    hht        smallint,   -- household type
    hht2       smallint,
    hupac      smallint,   -- presence and age of children
    noc        smallint,   -- own children in household
    npf        smallint,   -- family size
    r18        smallint,   -- any person under 18
    r60        smallint,
    r65        smallint,
    wif        smallint,   -- workers in family
    workstat   smallint,
    hhl        smallint,   -- household language
    lngi       smallint,   -- limited English household
    multg      smallint,
    partner    smallint,
    mv         smallint,   -- when moved in
    yrblt      smallint,
    vacs       smallint,   -- vacancy status (vacant units only)
    grpip      smallint,   -- gross rent as % of income
    ocpip      smallint,

    is_gq      boolean GENERATED ALWAYS AS (typehugq <> 1) STORED,
    occupied   boolean GENERATED ALWAYS AS (typehugq = 1 AND np > 0) STORED,

    PRIMARY KEY (serialno)
);

CREATE INDEX IF NOT EXISTS pums_hh_puma_ix
    ON census.pums_households (state, puma);

-- ── Persons ──────────────────────────────────────────────────────────────────
-- One row per person. serialno joins to the household record; a GQ person's
-- household row exists but carries no income or vehicles.
--
-- pernp (own earnings) is the column that fixes the model's standing income
-- problem: the survey reports HOUSEHOLD income, LODES segments on JOB
-- earnings, and the two are not the same quantity. A PUMS person carries both,
-- so an agent can hold a household income AND the personal earnings that
-- belong to it.

CREATE TABLE IF NOT EXISTS census.pums_persons (
    serialno   text     NOT NULL,
    sporder    smallint NOT NULL,
    state      char(2)  NOT NULL,
    puma       char(5)  NOT NULL,
    pwgtp      integer  NOT NULL,          -- person weight (nonzero for GQ too)
    adjinc     integer  NOT NULL,

    agep       smallint NOT NULL,          -- age in years
    sex        smallint,                   -- 1 male, 2 female
    relshipp   smallint,
    mar        smallint,
    cit        smallint,
    nativity   smallint,
    rac1p      smallint,
    hisp       smallint,
    lanx       smallint,
    eng        smallint,
    schl       smallint,                   -- educational attainment
    sch        smallint,                   -- school enrolment
    mil        smallint,

    esr        smallint,   -- employment status recode (1,2,4,5 = employed)
    wkhp       smallint,   -- usual hours worked per week
    wkl        smallint,   -- when last worked
    wkwn       smallint,   -- weeks worked
    indp       text,       -- industry code
    naicsp     text,       -- NAICS string
    occp       text,
    socp       text,
    powpuma    text,       -- place-of-work PUMA
    powsp      text,       -- place-of-work state
    jwtrns     smallint,   -- means of transportation to work
    jwmnp      smallint,   -- travel time to work, minutes

    wagp       integer,    -- wages, as reported
    semp       integer,    -- self-employment income, as reported
    pernp      integer,    -- total person EARNINGS, as reported
    pincp      integer,    -- total person income, as reported
    intp       integer,
    retp       integer,
    povpip     integer,    -- income-to-poverty ratio
    pernp_adj  double precision GENERATED ALWAYS AS
                   (pernp::double precision * adjinc / 1000000.0) STORED,
    pincp_adj  double precision GENERATED ALWAYS AS
                   (pincp::double precision * adjinc / 1000000.0) STORED,
    wagp_adj   double precision GENERATED ALWAYS AS
                   (wagp::double precision * adjinc / 1000000.0) STORED,

    dis        smallint,   -- disability recode (1 = with a disability)
    ddrs       smallint,   -- self-care difficulty
    dear       smallint,   -- hearing
    deye       smallint,   -- vision
    dout       smallint,   -- independent living
    dphy       smallint,   -- ambulatory  <- the ABM's mobility_needs signal
    drem       smallint,   -- cognitive

    PRIMARY KEY (serialno, sporder)
);

CREATE INDEX IF NOT EXISTS pums_person_puma_ix
    ON census.pums_persons (state, puma);

-- ── Tract -> PUMA crosswalk ──────────────────────────────────────────────────
-- The join that makes the two sources one pipeline: a LODES home block gives
-- an 11-digit tract, this gives the PUMA, and the PUMA selects the PUMS sample
-- that describes people living there. 2020 tracts to 2020 PUMAs, so it matches
-- both LODES8 2023 (2020 blocks) and ACS 2023 5-year (2020 geography).

CREATE TABLE IF NOT EXISTS census.tract_puma (
    tract_geoid char(11) NOT NULL,   -- state(2) + county(3) + tract(6)
    state       char(2)  NOT NULL,
    puma        char(5)  NOT NULL,
    PRIMARY KEY (tract_geoid)
);

CREATE INDEX IF NOT EXISTS tract_puma_puma_ix ON census.tract_puma (state, puma);

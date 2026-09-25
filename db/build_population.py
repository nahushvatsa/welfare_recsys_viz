#!/usr/bin/env python3
"""Build agent populations: LODES geography x reweighted PUMS x survey person.

    python3 db/build_population.py --metros dc --replicates 1 --agents 2000
    python3 db/build_population.py --all

Each agent is assembled from three sources, each used for what it alone can do:

1. **LODES** places it. A home-block -> work-block pair is drawn in proportion
   to the jobs on it, so commute geography follows the published distribution.
   The pair also fixes the job's age band, earnings band and industry sector.
2. **PUMS**, reweighted to the home tract (db/build_tract_weights.py), says who
   lives there. One real person is drawn from the tract's fitted sample subject
   to the bands LODES just fixed, bringing age, own earnings, household income,
   vehicles, disability and household size TOGETHER — the joint distribution
   the model needs and no aggregate table can supply.
3. **The survey** supplies psychometrics and platform attitudes, by matching a
   respondent on sex, employment, income band and age band.

WORKERS AND NON-WORKERS ARE PLACED DIFFERENTLY, and the difference is
deliberate. A worker's leisure trip departs from its WORKPLACE, which is always
in the core, a few km from the venues. A non-worker has no workplace, so its
trip departs from home (welfare_rs/agent.py, `origin_after_work`). Drawing
non-worker homes across the whole metro would put much of the population 30-60
km from every venue in the catalog and let those trips dominate every
distance-normalised metric — while describing a behaviour the model does not
represent, since a suburban non-worker's real options are suburban venues that
are not in the catalog. So non-workers are drawn from CORE blocks only: the
population is "people whose day is anchored in the core", either because they
work there or because they live there.

The worker share is not a parameter. It is jobs-in-core against core-resident
non-working adults, both measured.

Output is public.metro_population, read directly by the engine. Nothing here
happens at run time: a simulation reads N rows and builds agents from them.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
from collections import defaultdict

import numpy as np
import psycopg

DSN = os.environ.get("WELFARE_LOADER_DSN", "")
BUILD_VERSION = 1          # bump to invalidate populations built by older logic

# ── Recodes shared with the survey's own band vocabulary ─────────────────────
# Survey Q1 (age) and Q6 (household income) are ordinal bands; an agent has to
# land in the same bands to be matchable. Values verified against
# survey.personas: 473 respondents, 6 age bands, 7 income bands.
SURVEY_AGE_BANDS = [
    ("18-24", 18, 24), ("25-34", 25, 34), ("35-44", 35, 44),
    ("45-54", 45, 54), ("55-64", 55, 64), ("65 or older", 65, 200),
]
SURVEY_INCOME_BANDS = [
    ("Under $25,000", None, 25000),
    ("$25,000 - $49,999", 25000, 50000),
    ("$50,000 - $74,999", 50000, 75000),
    ("$75,000 - $99,999", 75000, 100000),
    ("$100,000 - $149,000", 100000, 150000),
    ("$150,000 - $199,999", 150000, 200000),
    ("$200,000 or more", 200000, None),
]
SURVEY_EMPLOYED = {"Employed full-time", "Employed part-time", "Self-employed"}

# LODES segments (tech doc, LODES8 rev. 20241105).
#   SA01 age 14-29, SA02 30-54, SA03 55-99
#   SE01 <=$1250/month, SE02 $1251-3333, SE03 >$3333
#   SI01 goods producing: NAICS 11, 21, 23, 31-33
#   SI02 trade/transport/utilities: NAICS 22, 42, 44-45, 48-49
#   SI03 all other services: the remaining twelve sectors
SI01_SECTORS = {"11", "21", "23", "31", "32", "33"}
SI02_SECTORS = {"22", "42", "44", "45", "48", "49"}

# ── Car-commute plausibility ────────────────────────────────────────────────
# LODES records WHERE people commute, never HOW. When this cap was set the model
# drove every one of them (car-only), and the OD tail contains commutes nobody
# drives daily: the longest NYC pair is 488 km, and 9.5% of NYC jobs (16.7% in
# LA) sit on pairs over 50 km.
#
# The cap is set from what car commuters actually report. PUMS JWMNP for people
# whose mode is car (JWTRNS = 1): New Jersey median 25 min, p90 60, p99 134;
# New York median 20, p90 55, p99 138. So 135 minutes one way is the 99th
# percentile of real car commuting, and a pair implying more than that is
# excluded from the draw rather than resampled — it is simply not a commute
# anyone makes by car.
#
# Distance is converted to time with an explicit, crude pair of assumptions,
# because the builder has no road graph: a 1.25 detour factor over the
# straight line, and 55 km/h effective door-to-door speed. 135 min then means
# ~99 km straight-line. The cap only trims a tail, so its exact placement
# matters far less than removing the impossible part of it.
MAX_CAR_COMMUTE_MIN = 135.0
DETOUR_FACTOR = 1.25
EFFECTIVE_SPEED_KMH = 55.0

# NOTE this does NOT make every remaining commute a plausible CAR commute.
# New Brunswick to Manhattan is ~55 km, about 75 minutes driving — within what
# people report — yet 62.6% of New Jersey residents working in New York take
# transit and only 35.4% drive. Distance cannot separate those; mode can. Every
# agent carries its own PUMS mode (commute_mode / JWTRNS), and the draw below
# now matches on it, but the model has no transit mode to put those riders on:
# outside the core they still drive (or, in NYC, take a ride if carless). That
# is a stated limitation rather than something this cap fixes.

# Plausible DOOR-TO-DOOR speed ranges by PUMS mode (JWTRNS), in network km/h.
# A person can hold a commute pair if their REPORTED commute time and the
# pair's distance fit some speed inside their mode's range. Ranges, not point
# speeds, because door-to-door speed is not one number: a 1 km drive (walk to
# the car, lights, parking) averages a fraction of a 30 km freeway commute, and
# a transit trip's wait matters more the shorter it is. These are stated
# assumptions, deliberately wide, not estimates:
#   walk 3.5-6      walking pace, ~1.0-1.6 m/s
#   bike 10-20      urban cycling including stops
#   car 8-60        dense-city short trip up to a freeway commute (the 55 km/h
#                   the car cap uses sits inside)
#   transit 8-50    local bus or short subway ride with its wait, up to commuter
#                   rail
# Codes: 1 car/truck/van, 2 bus, 3 subway, 4 commuter rail, 5 light rail,
# 6 ferry, 7 taxi, 8 motorcycle, 9 bicycle, 10 walked. Worked from home (11),
# other (12) and a missing mode or time carry no range: those people are
# matched on the remaining keys only.
MODE_SPEED_RANGE_KMH = {
    1: (8.0, 60.0), 7: (8.0, 60.0), 8: (8.0, 60.0),
    2: (8.0, 50.0), 3: (8.0, 50.0), 4: (8.0, 50.0), 5: (8.0, 50.0), 6: (8.0, 50.0),
    9: (10.0, 20.0),
    10: (3.5, 6.0),
}

# Commute-DISTANCE bands (straight-line km). The drawn pair falls in one; a
# person is filed under every band their reported time and mode make
# plausible, and the draw asks for the pair's band.
#
# This replaces matching on TIME with every pair converted at car speed. That
# made a 3 km Manhattan commute "4 minutes", matchable only to people reporting
# under 15 — walkers and very short drives — and put 74% of NYC's core
# commuters on foot against 24.5% in ACS B08301 for the same tracts. A single
# speed per mode fails the other way: at 55 km/h no driver reports a time short
# enough for a 1 km pair, so only walkers qualified (Miami walk share rose from
# 21% to 28%). Hence the ranges above.
COMMUTE_DIST_BANDS_KM = [(0, 0.8), (0.8, 1.6), (1.6, 3), (3, 6), (6, 12),
                         (12, 25), (25, 100000)]


def commute_dist_band(km):
    """Index of the commute-distance band a pair falls in, or None if unknown."""
    if km is None:
        return None
    for i, (lo, hi) in enumerate(COMMUTE_DIST_BANDS_KM):
        if lo <= km < hi:
            return i
    return len(COMMUTE_DIST_BANDS_KM) - 1


def plausible_dist_bands(minutes, mode):
    """Distance bands a reported commute time makes plausible for this mode.

    Straight-line km between ``minutes * v_lo`` and ``minutes * v_hi`` (over the
    detour factor). Returns a tuple of band indices, or None when the person
    carries no range (see MODE_SPEED_RANGE_KMH).
    """
    if minutes is None or mode is None:
        return None
    rng = MODE_SPEED_RANGE_KMH.get(int(mode))
    if rng is None:
        return None
    lo_km = float(minutes) / 60.0 * rng[0] / DETOUR_FACTOR
    hi_km = float(minutes) / 60.0 * rng[1] / DETOUR_FACTOR
    return tuple(i for i, (b_lo, b_hi) in enumerate(COMMUTE_DIST_BANDS_KM)
                 if b_lo < hi_km and b_hi > lo_km)


def implied_car_minutes(km):
    """One-way driving time implied by a straight-line distance."""
    return km * DETOUR_FACTOR / EFFECTIVE_SPEED_KMH * 60.0


def age_segment(age):
    return 0 if age <= 29 else (1 if age <= 54 else 2)


def earnings_segment(annual):
    """LODES earnings band from an annual figure. LODES segments MONTHLY job
    earnings, so the PUMS annual figure is divided by 12. PERNP is earnings
    from all jobs while LODES counts one job, which is the residual mismatch
    this whole build reduces but cannot remove."""
    if annual is None:
        return None
    monthly = annual / 12.0
    return 0 if monthly <= 1250 else (1 if monthly <= 3333 else 2)


def industry_segment(naicsp):
    """LODES industry segment from a PUMS NAICSP code.

    NAICSP is mostly numeric ('3364', '22S'), but carries a few aggregate codes
    whose first character is the only usable digit ('3MS' = manufacturing,
    '4MS' = wholesale/retail). Every 3x sector is SI01 and every 4x sector is
    SI02, so the first digit resolves those unambiguously.
    """
    if not naicsp:
        return None
    head = naicsp[:2]
    if head.isdigit():
        if head in SI01_SECTORS:
            return 0
        if head in SI02_SECTORS:
            return 1
        return 2
    first = naicsp[0]
    if first == "3":
        return 0
    if first == "4":
        return 1
    return 2


def survey_age_band(age):
    for name, lo, hi in SURVEY_AGE_BANDS:
        if lo <= age <= hi:
            return name
    return None


def survey_income_band(income):
    if income is None:
        return None
    for name, lo, hi in SURVEY_INCOME_BANDS:
        if (lo is None or income >= lo) and (hi is None or income < hi):
            return name
    return None


def rng_for(metro, replicate):
    """Deterministic per (metro, replicate) stream.

    Seeded from a STRING digest rather than an arithmetic combination: adjacent
    integer seeds give correlated first draws, and populations are built in
    replicate order, so that correlation would run straight through the
    replicates the seeds are supposed to make independent.
    """
    key = f"pop:{BUILD_VERSION}:{metro}:{replicate}".encode()
    return np.random.default_rng(int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "big"))


# ── Survey respondents, bucketed for matching ────────────────────────────────

# Matching keys, most specific first; each level drops the next key in the
# user's order: sex, then employment, then income, then age. Age is the last
# thing given up.
#
# MORE KEYS IS NOT AUTOMATICALLY BETTER. The survey holds 473 respondents, so
# the four-key cells average 3.5 people and 32 of them hold exactly one — whose
# psychometrics then get stamped onto every agent landing there. Measured on a
# 2,000-agent DC build: 412 respondents used, the most-used 72 times, effective
# sample size 133, and the trust x autonomy correlation attenuated from -0.515
# to -0.401. Dropping sex as a key gives 75 cells averaging 6.3 respondents
# with 9 singletons. `--match-keys` exposes the trade-off rather than burying it.
DEFAULT_MATCH_KEYS = ("sex", "employment", "income", "age")


def build_match_levels(keys):
    """Ladder of key subsets, dropping one key at a time from the front."""
    keys = tuple(keys)
    return [(len(keys) - i, keys[i:]) for i in range(len(keys) + 1)]


MATCH_LEVELS = build_match_levels(DEFAULT_MATCH_KEYS)


def set_match_keys(keys):
    """Replace the matching ladder (used by --match-keys)."""
    global MATCH_LEVELS
    MATCH_LEVELS = build_match_levels(keys)


def load_survey_buckets(cur):
    """Respondent ids bucketed by every key combination the ladder can use.

    Drop order is the user's: sex first, then employment, then income, then
    age. Age is the last key given up, so an agent's psychometrics always come
    from someone of roughly its own age unless the survey has nobody at all.
    """
    cur.execute('SELECT "PersonaID", "Sex", "Employment", "Income", "Age" FROM survey.personas')
    rows = cur.fetchall()
    if not rows:
        raise SystemExit("FATAL: survey.personas returned no rows")
    buckets = {lvl: defaultdict(list) for lvl, _ in MATCH_LEVELS}
    for pid, sex, employment, income, age in rows:
        employed = employment in SURVEY_EMPLOYED
        full = {"sex": sex, "employment": employed, "income": income, "age": age}
        for lvl, keys in MATCH_LEVELS:
            buckets[lvl][tuple(full[k] for k in keys)].append(pid)
    return buckets


def match_respondent(buckets, rng, *, sex, employed, income_band, age_band):
    """Draw a respondent, giving up keys in the agreed order until one matches."""
    full = {"sex": sex, "employment": employed, "income": income_band, "age": age_band}
    for lvl, keys in MATCH_LEVELS:
        pool = buckets[lvl].get(tuple(full[k] for k in keys))
        if pool:
            return pool[int(rng.integers(len(pool)))], lvl, "+".join(keys) or "none"
    raise SystemExit("FATAL: no survey respondent matched even at level 0")


# ── PUMS pools ───────────────────────────────────────────────────────────────

_EMPTY_INDEX = np.zeros(0, dtype=np.int64)


class PumaSample:
    """One PUMA's adult PUMS records, pre-recoded into the bands used here."""

    def __init__(self, cur, state, puma):
        cur.execute("""
            SELECT p.serialno, p.sporder, p.agep, p.sex, p.esr, p.pernp_adj,
                   p.dis, p.dphy, p.jwtrns, p.jwmnp, p.naicsp,
                   h.hincp_adj, h.veh, h.np, h.wgtp
            FROM census.pums_persons p
            JOIN census.pums_households h USING (serialno)
            WHERE p.state = %s AND p.puma = %s AND p.agep >= 18
        """, (state, puma))
        self.rows = []
        for (serialno, sporder, agep, sex, esr, pernp, dis, dphy, jwtrns, jwmnp,
             naicsp, hincp, veh, np_, wgtp) in cur.fetchall():
            employed = esr in (1, 2, 4, 5)
            self.rows.append({
                "serialno": serialno, "sporder": sporder, "agep": agep, "sex": sex,
                "employed": employed, "pernp": pernp, "hincp": hincp, "veh": veh,
                "dis": (None if dis is None else dis == 1), "np": np_,
                "ambulatory": (None if dphy is None else dphy == 1),
                "jwtrns": jwtrns, "jwmnp": jwmnp, "naicsp": naicsp,
                "age_seg": age_segment(agep),
                "earn_seg": earnings_segment(pernp) if employed else None,
                "ind_seg": industry_segment(naicsp) if employed else None,
                "dist_bands": plausible_dist_bands(jwmnp, jwtrns) if employed else None,
                "base_w": float(wgtp or 0.0),
            })
        self.serial_index = defaultdict(list)
        for i, r in enumerate(self.rows):
            self.serial_index[r["serialno"]].append(i)

        # Candidate indices per draw key, computed ONCE for the PUMA. Every
        # tract in this PUMA reuses them and only its own weights differ, which
        # turns the per-tract work into a numpy slice. Filtering per tract
        # instead would be a Python pass over the whole sample for every tract
        # and every key — hundreds of millions of iterations across a build.
        # PUMS's own household weights, used only when a tract's fitted
        # weights leave a draw with nothing to choose from.
        self.base_w = np.array([r["base_w"] for r in self.rows], dtype=np.float64)
        if self.base_w.sum() <= 0:
            self.base_w = np.ones(len(self.rows), dtype=np.float64)

        self.index = {}
        buckets = defaultdict(list)
        for i, r in enumerate(self.rows):
            if r["employed"]:
                a, e, ind = r["age_seg"], r["earn_seg"], r["ind_seg"]
                # Exactly the keys the draw ladder asks for, no more. A person
                # sits under every distance band their commute makes plausible;
                # one with no range (worked from home, other, missing) only
                # under the keys that ignore distance, as before.
                for t in (r["dist_bands"] or ()):
                    buckets[(True, (a, e, ind, t))].append(i)
                    buckets[(True, (a, e, None, t))].append(i)
                    buckets[(True, (a, None, None, t))].append(i)
                buckets[(True, (a, e, ind, None))].append(i)
                buckets[(True, (a, e, None, None))].append(i)
                buckets[(True, (a, None, None, None))].append(i)
                buckets[(True, None)].append(i)
            else:
                buckets[(False, None)].append(i)
        for key, idx in buckets.items():
            self.index[key] = np.asarray(idx, dtype=np.int64)

    def candidates(self, key):
        return self.index.get(key, _EMPTY_INDEX)

    def __len__(self):
        return len(self.rows)


class TractPool:
    """The PUMA's people, weighted for one tract, indexed by draw key.

    Keys, in the order the worker draw gives them up:
      (age, earn, ind) -> (age, earn) -> (age,) -> () for workers,
      () for non-workers.
    Each pool is a (row indices, cumulative weights) pair, built on first use:
    a tract is drawn from a few hundred times at most, so building every key up
    front would cost more than it saves.
    """

    def __init__(self, sample: PumaSample, weights: dict):
        self.sample = sample
        self.w = np.zeros(len(sample.rows), dtype=np.float64)
        for serialno, weight in weights.items():
            for i in sample.serial_index.get(serialno, ()):
                self.w[i] = weight
        self._cache = {}
        self._base_cache = {}

    def _pool(self, key, base=False):
        cache = self._base_cache if base else self._cache
        cached = cache.get(key)
        if cached is not None:
            return cached
        candidates = self.sample.candidates(key)
        if candidates.size:
            weights = (self.sample.base_w if base else self.w)[candidates]
            keep = weights > 0
            arr = candidates[keep]
            cum = np.cumsum(weights[keep])
        else:
            arr, cum = _EMPTY_INDEX, np.zeros(0)
        entry = (arr, cum)
        cache[key] = entry
        return entry

    def draw(self, rng, *, worker, age_seg=None, earn_seg=None, ind_seg=None,
             dist_band=None):
        """Draw one person, relaxing the bands only as far as needed.

        Industry and earnings are given up before the commute-distance band,
        because distance is what keeps the drawn geography honest: a 40 km pair
        should belong to someone whose own reported commute reaches that far,
        while industry is only a sharpener.
        """
        if worker:
            ladder = [
                (age_seg, earn_seg, ind_seg, dist_band),
                (age_seg, earn_seg, None, dist_band),
                (age_seg, None, None, dist_band),
                (age_seg, earn_seg, ind_seg, None),
                (age_seg, earn_seg, None, None),
                (age_seg, None, None, None),
                None,
            ]
        else:
            ladder = [None]
        for segs in ladder:
            arr, cum = self._pool((worker, segs))
            if arr.size and cum[-1] > 0:
                j = int(np.searchsorted(cum, rng.random() * cum[-1], side="right"))
                return self.sample.rows[int(arr[min(j, arr.size - 1)])], False

        # Nothing in this tract's fitted weights can serve the draw. It happens
        # where ACS and LODES disagree about a tract: the employment control
        # says essentially nobody employed lives there, so the fit drove every
        # household containing a worker to ~0, while LODES still places a
        # commuter's home in it. The agent keeps the geography LODES gave it and
        # takes its demographics from the PUMA's own PUMS weights instead of the
        # tract's. Counted and reported per metro, never silent.
        for segs in ladder:
            arr, cum = self._pool((worker, segs), base=True)
            if arr.size and cum[-1] > 0:
                j = int(np.searchsorted(cum, rng.random() * cum[-1], side="right"))
                return self.sample.rows[int(arr[min(j, arr.size - 1)])], True
        return None, None


# ── Metro inputs ─────────────────────────────────────────────────────────────

def load_pairs(cur, metro):
    cur.execute("""
        SELECT h_geoid, w_geoid, h_lat, h_lon, w_lat, w_lon, jobs,
               sa01, sa02, sa03, se01, se02, se03, si01, si02, si03
        FROM public.metro_od_pairs WHERE metro = %s ORDER BY h_geoid, w_geoid
    """, (metro,))
    rows = cur.fetchall()
    if not rows:
        raise SystemExit(f"FATAL: no commute pairs for metro {metro}")
    h_geoid = np.array([r[0] for r in rows])
    w_geoid = np.array([r[1] for r in rows])
    coords = np.array([[r[2], r[3], r[4], r[5]] for r in rows], dtype=np.float64)
    jobs = np.array([r[6] for r in rows], dtype=np.float64)
    # Straight-line home->work distance, used for the car-commute cap and for
    # the commute-distance band each drawn pair is matched on.
    lat1, lon1, lat2, lon2 = (np.radians(coords[:, 0]), np.radians(coords[:, 1]),
                              np.radians(coords[:, 2]), np.radians(coords[:, 3]))
    km = 6371.0 * 2 * np.arcsin(np.sqrt(
        np.sin((lat2 - lat1) / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2))
    minutes = implied_car_minutes(km)
    seg = np.array([[r[7] or 0, r[8] or 0, r[9] or 0,
                     r[10] or 0, r[11] or 0, r[12] or 0,
                     r[13] or 0, r[14] or 0, r[15] or 0] for r in rows], dtype=np.float64)
    return h_geoid, w_geoid, coords, jobs, seg, minutes, km


def load_core_homes(cur, metro):
    """Core blocks with their expected count of non-working adults.

    pop20 counts everyone, including children and the employed, so it is the
    wrong weight on its own: it would place non-workers in proportion to total
    population rather than to the people the category describes. Each block is
    scaled by its tract's rate of 16+ adults not employed (ACS B23025
    unemployed + not-in-labour-force over B01001 total population).
    """
    cur.execute("""
        WITH rate AS (
            SELECT e.geoid AS tract,
                   sum(e.estimate) FILTER (WHERE e.table_id='b23025' AND e.varnum IN (5,7))
                     / NULLIF(max(e.estimate) FILTER (WHERE e.table_id='b01001' AND e.varnum=1), 0)
                     AS nonworker_share
            FROM census.acs_estimates e
            WHERE e.sumlevel = 140
              AND ((e.table_id='b23025' AND e.varnum IN (5,7))
                   OR (e.table_id='b01001' AND e.varnum=1))
            GROUP BY e.geoid
        )
        SELECT hb.geoid, hb.tract_geoid, hb.lat, hb.lon,
               hb.pop20 * COALESCE(r.nonworker_share, 0.35) AS expected_nonworkers
        FROM public.metro_home_blocks hb
        LEFT JOIN rate r ON r.tract = hb.tract_geoid
        WHERE hb.metro = %s
        ORDER BY hb.geoid
    """, (metro,))
    rows = cur.fetchall()
    if not rows:
        raise SystemExit(f"FATAL: no core home blocks for metro {metro} — "
                         "run db/build_tract_weights.py --what home-blocks first")
    geoid = np.array([r[0] for r in rows])
    tract = np.array([r[1] for r in rows])
    coords = np.array([[r[2], r[3]] for r in rows], dtype=np.float64)
    weight = np.array([float(r[4] or 0.0) for r in rows], dtype=np.float64)
    return geoid, tract, coords, weight


def draw_categorical(rng, weights, size):
    cum = np.cumsum(weights)
    return np.searchsorted(cum, rng.random(size) * cum[-1], side="right").clip(0, len(weights) - 1)


def build_metro(conn, metro, replicates, n_agents, log=print):
    cur = conn.cursor()
    h_geoid, w_geoid, pair_xy, jobs, seg, pair_minutes, pair_km = load_pairs(cur, metro)

    # Exclude commutes nobody drives daily. Zeroing the job weight removes the
    # pair from the draw entirely, which is what "not a car commute" means --
    # resampling would only slow the draw down to reach the same place.
    # Tracts ACS says nobody lives in. 127 of them (airports, parks, industrial
    # land) carry 0.07% of LODES jobs, which is noise infusion and edge cases
    # rather than residents: with zero households and zero population there is
    # nothing to fit, so the fit produced no weights and no person can be drawn
    # there. Their pairs leave the draw for the same reason as an impossible
    # commute — not because the model dislikes them, but because the home does
    # not exist.
    cur.execute("SELECT tract_geoid FROM census.pums_tract_weight_meta WHERE n_units > 0")
    habitable = {r[0] for r in cur.fetchall()}
    uninhabited = np.array([str(g)[:11] not in habitable for g in h_geoid])
    if uninhabited.any():
        log(f"  {metro}: {jobs[uninhabited].sum():,.0f} jobs "
            f"({jobs[uninhabited].sum() / jobs.sum():.2%}) have home tracts with no "
            f"residents in ACS — excluded")
        jobs = np.where(uninhabited, 0.0, jobs)

    too_far = pair_minutes > MAX_CAR_COMMUTE_MIN
    dropped_jobs = float(jobs[too_far].sum())
    total_before = float(jobs.sum())
    jobs = np.where(too_far, 0.0, jobs)
    log(f"  {metro}: car-commute cap at {MAX_CAR_COMMUTE_MIN:.0f} min one way "
        f"(~{MAX_CAR_COMMUTE_MIN * EFFECTIVE_SPEED_KMH / 60 / DETOUR_FACTOR:.0f} km "
        f"straight line) removes {dropped_jobs:,.0f} of {total_before:,.0f} jobs "
        f"({dropped_jobs / total_before:.1%}), longest kept "
        f"{pair_minutes[~too_far].max():.0f} min")
    c_geoid, c_tract, c_xy, c_weight = load_core_homes(cur, metro)

    unfit_blocks = np.array([t not in habitable for t in c_tract])
    if unfit_blocks.any():
        log(f"  {metro}: {int(unfit_blocks.sum()):,} core blocks sit in tracts with "
            f"no fitted weights — excluded from the non-worker draw")
        c_weight = np.where(unfit_blocks, 0.0, c_weight)

    total_jobs = float(jobs.sum())          # after the cap
    total_nonworkers = float(c_weight.sum())
    p_worker = total_jobs / (total_jobs + total_nonworkers)
    log(f"  {metro}: {total_jobs:,.0f} core jobs, {total_nonworkers:,.0f} core "
        f"non-working adults -> {p_worker:.1%} of agents are workers")

    cur.execute("SELECT tract_geoid, state, puma FROM census.tract_puma")
    puma_of = {t: (s, p) for t, s, p in cur.fetchall()}

    # Transit-commute share per tract (ACS B08301 var 10 over var 1). A
    # neighbourhood service measure, carried onto every agent living there.
    cur.execute('''
        SELECT geoid,
               sum(estimate) FILTER (WHERE varnum = 10)
                 / NULLIF(max(estimate) FILTER (WHERE varnum = 1), 0)
        FROM census.acs_estimates
        WHERE table_id = 'b08301' AND sumlevel = 140 AND varnum IN (1, 10)
        GROUP BY geoid
    ''')
    transit_share = {t: (float(v) if v is not None else None)
                     for t, v in cur.fetchall()}
    buckets = load_survey_buckets(cur)

    # All replicates are drawn first and then processed together, grouped by
    # PUMA. Each PUMA's PUMS sample and each tract's fitted weights are
    # expensive to build and identical across replicates, so loading them once
    # per metro instead of once per replicate is the difference between minutes
    # and hours. Determinism is unaffected: every replicate keeps its own RNG
    # stream and the grouping order is fixed.
    written = 0
    rngs = {r: rng_for(metro, r) for r in range(replicates)}
    agents = []
    t0 = time.time()
    for replicate in range(replicates):
        rng = rngs[replicate]

        is_worker = rng.random(n_agents) < p_worker
        n_w = int(is_worker.sum())
        pair_idx = draw_categorical(rng, jobs, n_w)
        block_idx = draw_categorical(rng, c_weight, n_agents - n_w)

        # The job's own segments, drawn from the pair's three marginals. LODES
        # publishes them separately and never their cross-tabulation, so they
        # are drawn independently within the pair — exact for the ~80% of jobs
        # that sit on a pair carrying a single job, an assumption only for the
        # rest.
        wi = bi = 0
        for i in range(n_agents):
            if bool(is_worker[i]):
                pi = int(pair_idx[wi]); wi += 1
                j = jobs[pi]
                segs = tuple(int(np.searchsorted(np.cumsum(seg[pi, k:k + 3]),
                                                 rng.random() * j, side="right").clip(0, 2))
                             for k in (0, 3, 6))
                tract = str(h_geoid[pi])[:11]
                agents.append({"rep": replicate, "i": i, "worker": True, "pair": pi,
                               "tract": tract, "age_seg": segs[0],
                               "earn_seg": segs[1], "ind_seg": segs[2],
                               "dist_band": commute_dist_band(float(pair_km[pi]))})
            else:
                ci = int(block_idx[bi]); bi += 1
                agents.append({"rep": replicate, "i": i, "worker": False,
                               "block": ci, "tract": str(c_tract[ci])})

    log(f"    drew {len(agents):,} agent geographies in {time.time() - t0:.0f}s")

    # Group by PUMA so each PUMS sample and its tract weights load once.
    by_puma = defaultdict(list)
    for a in agents:
        key = puma_of.get(a["tract"])
        if key is None:
            raise SystemExit(f"FATAL: tract {a['tract']} has no PUMA")
        by_puma[key].append(a)

    rows_by_rep = defaultdict(list)
    n_puma_fallback = 0
    t0 = time.time()
    for (state, puma), group in sorted(by_puma.items()):
        sample = PumaSample(cur, state, puma)
        if not len(sample):
            raise SystemExit(f"FATAL: PUMA {state}-{puma} has no adult PUMS records")
        tracts = sorted({a["tract"] for a in group})
        cur.execute("SELECT tract_geoid, serialno, weight FROM census.pums_tract_weights "
                    "WHERE tract_geoid = ANY(%s)", (tracts,))
        w_by_tract = defaultdict(dict)
        for tract, serialno, weight in cur.fetchall():
            w_by_tract[tract][serialno] = weight

        pools = {}
        for a in group:
            pool = pools.get(a["tract"])
            if pool is None:
                weights = w_by_tract.get(a["tract"])
                if not weights:
                    raise SystemExit(
                        f"FATAL: tract {a['tract']} has no fitted weights — "
                        "run db/build_tract_weights.py --what fit")
                pool = pools[a["tract"]] = TractPool(sample, weights)

            rng = rngs[a["rep"]]
            person, used_puma = pool.draw(
                rng, worker=a["worker"],
                age_seg=a.get("age_seg"), earn_seg=a.get("earn_seg"),
                ind_seg=a.get("ind_seg"), dist_band=a.get("dist_band"))
            if person is None:
                raise SystemExit(
                    f"FATAL: PUMA {state}-{puma} has no "
                    f"{'worker' if a['worker'] else 'non-worker'} at all — not a "
                    "tract-level problem, the sample itself is empty")
            n_puma_fallback += int(used_puma)

            sex_label = ("Male" if person["sex"] == 1
                         else "Female" if person["sex"] == 2 else None)
            age_band = survey_age_band(person["agep"])
            income_band = survey_income_band(person["hincp"])
            pid, level, keys = match_respondent(
                buckets, rng, sex=sex_label, employed=person["employed"],
                income_band=income_band, age_band=age_band)

            if a["worker"]:
                pi = a["pair"]
                home_geoid, work_geoid = str(h_geoid[pi]), str(w_geoid[pi])
                hlat, hlon, wlat, wlon = pair_xy[pi]
                ind_seg = a["ind_seg"] + 1
            else:
                ci = a["block"]
                home_geoid, work_geoid = str(c_geoid[ci]), None
                hlat, hlon = c_xy[ci]
                wlat = wlon = None
                ind_seg = None

            rows_by_rep[a["rep"]].append((
                metro, a["rep"], a["i"], a["worker"],
                home_geoid, a["tract"], float(hlat), float(hlon),
                work_geoid,
                None if wlat is None else float(wlat),
                None if wlon is None else float(wlon),
                puma, person["serialno"], person["sporder"],
                person["agep"], person["sex"], person["employed"], ind_seg,
                person["pernp"], person["hincp"], person["veh"],
                person["dis"], person["ambulatory"], person["np"],
                person["jwtrns"],
                float(pair_minutes[a["pair"]]) if a["worker"] else None,
                person["jwmnp"], transit_share.get(a["tract"]),
                pid, keys, level,
            ))

    log(f"    matched {len(agents):,} agents to PUMS + survey in "
        f"{time.time() - t0:.0f}s")
    if n_puma_fallback:
        log(f"    {n_puma_fallback:,} agents ({n_puma_fallback / len(agents):.2%}) took "
            f"PUMA-level weights: their tract's fit had nobody of that kind")

    for replicate in sorted(rows_by_rep):
        rows = rows_by_rep[replicate]
        rows.sort(key=lambda r: r[2])
        cur.execute("DELETE FROM public.metro_population WHERE metro = %s AND replicate = %s",
                    (metro, replicate))
        with cur.copy("""
            COPY public.metro_population
                (metro, replicate, agent_id, is_worker,
                 home_geoid, home_tract, home_lat, home_lon,
                 work_geoid, work_lat, work_lon, puma,
                 serialno, sporder, age, sex, employed, industry_seg,
                 own_earnings, hh_income, vehicles, disability, ambulatory,
                 hh_size, commute_mode, commute_min_implied,
                 commute_min_reported, transit_share,
                 respondent_id, match_keys, match_level)
            FROM STDIN
        """) as cp:
            for row in rows:
                cp.write_row(row)
        conn.commit()
        written += len(rows)
        n_workers = sum(1 for r in rows if r[3])
        log(f"    replicate {replicate}: {len(rows):,} agents written "
            f"({n_workers:,} workers, {len(rows) - n_workers:,} non-workers)")
    return written


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metros", nargs="*", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--replicates", type=int, default=12)
    ap.add_argument("--agents", type=int, default=20000)
    ap.add_argument("--match-keys", default=",".join(DEFAULT_MATCH_KEYS),
                    help="comma-separated survey matching keys, most specific "
                         "first; each is dropped in turn when a cell is empty")
    args = ap.parse_args()

    keys = tuple(k.strip() for k in args.match_keys.split(",") if k.strip())
    unknown = [k for k in keys if k not in DEFAULT_MATCH_KEYS]
    if unknown:
        raise SystemExit(f"FATAL: unknown match keys {unknown}; "
                         f"known: {list(DEFAULT_MATCH_KEYS)}")
    set_match_keys(keys)
    print(f"survey matching on {' + '.join(keys)}")

    with psycopg.connect(DSN) as conn:
        conn.cursor().execute("SET work_mem = '1GB'")
        cur = conn.cursor()
        if args.all or not args.metros:
            cur.execute("SELECT DISTINCT metro FROM public.metro_od_pairs ORDER BY metro")
            metros = [r[0] for r in cur.fetchall()]
        else:
            metros = args.metros
        print(f"building {args.agents:,} agents x {args.replicates} replicates "
              f"for {len(metros)} metro(s)", flush=True)
        total = 0
        for metro in metros:
            total += build_metro(conn, metro, args.replicates, args.agents)
        print(f"done: {total:,} agent rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())

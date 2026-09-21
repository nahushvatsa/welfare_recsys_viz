#!/usr/bin/env python3
"""Fit the PUMS sample to each tract's ACS control totals (IPU).

    python3 db/build_tract_weights.py --what validate    # check the recodes first
    python3 db/build_tract_weights.py --what fit         # fit every needed tract
    python3 db/build_tract_weights.py --what fit --limit 50

THE PROBLEM. PUMS describes people jointly — one person's age, earnings,
vehicles and household composition together — but locates them only to a PUMA
of 100k+ people (~34 tracts). ACS locates precisely but publishes one margin at
a time: how many households have no vehicle, how many people are 25-34, never
which people are which. Neither alone can place a realistic person in a tract.

THE METHOD. Iterative Proportional Updating (Ye et al. 2009, the method behind
PopGen / ActivitySim). Start from each PUMS household's own weight, then cycle
through the controls, scaling the weights that contribute to each control so
its weighted total matches the tract's published figure. Repeat until every
control is matched or the sample cannot do better. The household stays intact
throughout — only how many of it the tract is thought to contain changes — so
the joint structure survives while the marginals become local.

IPU rather than plain IPF because the controls are of two kinds: households
(income, vehicles, size) and persons (age, sex, employment, disability,
commute). One weight per household has to satisfy both at once, which is
exactly what IPU does and what fitting households and persons separately
cannot.

WHY TRACTS AND NOT BLOCK GROUPS. Measured on Manhattan's block groups: ~600
households each, +/-33% margin of error on that total, and 134% on a single
income bucket. Fitting to numbers that noisy fits noise. Tracts are ~4x larger.

UNIVERSES ARE NOT DECORATION. Each ACS table counts a specific population --
households vs people, occupied units vs all units, 16+ vs everyone, civilian
noninstitutionalised vs all. A PUMS recode that includes the wrong people
produces a control that cannot be matched and quietly distorts every weight in
the tract. `--what validate` checks each recode by summing PUMS weights for a
whole state and comparing with the same ACS control summed over that state's
counties; a recode whose universe is wrong shows up there as a percentage gap.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from collections import defaultdict

import numpy as np
import psycopg

DSN = os.environ.get("WELFARE_LOADER_DSN", "")

# Sweeps. More does not mean better here: ACS's own tract tables contradict
# each other (for tract 36047003000 the household-size table implies at least
# 1,954 people in households while the population table reports 1,818 — a 7%
# contradiction between two published tables), so no weighting satisfies every
# control and the sweeps oscillate around the compromise. Measured on the worst
# tracts, 200 sweeps land no closer than 50 and sometimes further, which is why
# the best sweep is kept rather than the last.
MAX_ITER = 50

# Convergence. Relative error, but floored: a control of 4 bike commuters in
# one tract cannot be matched to a fraction of a percent, and ACS itself
# publishes such cells with margins of error of 50-130%, so demanding
# precision there is chasing noise and would report every tract as failed.
# Errors are measured against max(target, ERROR_FLOOR), which makes the
# criterion "meet the controls that carry real people".
TOL = 0.05          # 5% of a control, or of ERROR_FLOOR for small ones
ERROR_FLOOR = 25.0  # households/people below which a cell is noise
WEIGHT_EPS = 1e-7   # stop when a full sweep stops moving the weights
ZERO_TARGET = 1e-9  # a control of 0 drives its contributors to ~0, not exactly 0
MIN_WEIGHT = 1e-6   # below this a weight is not stored

STATE_OF = {}       # filled per run: tract -> state fips


# ── Control definitions ──────────────────────────────────────────────────────
# Each control: (name, level, acs_table, [acs varnums summed], pums test)
#
# level 'hh'     — the test takes a household dict, returns True/False.
#                  Universe: occupied housing units (never group quarters,
#                  never vacant), because that is what the ACS household
#                  tables count.
# level 'person' — the test takes a person dict, returns True/False. Incidence
#                  is the COUNT of matching people in the household, which is
#                  what ties a household weight to a person control.

AGE_BANDS = [(0, 17), (18, 24), (25, 34), (35, 44), (45, 54), (55, 64), (65, 200)]

# B01001: 001 total, 002 male total, 003-025 male bands, 026 female total,
# 027-049 female bands (female band = male band + 24).
B01001_MALE = {
    (0, 17): [3, 4, 5, 6],
    (18, 24): [7, 8, 9, 10],
    (25, 34): [11, 12],
    (35, 44): [13, 14],
    (45, 54): [15, 16],
    (55, 64): [17, 18, 19],
    (65, 200): [20, 21, 22, 23, 24, 25],
}

# B18101 "with a disability" cells, male then female.
B18101_WITH = [4, 7, 10, 13, 16, 19, 23, 26, 29, 32, 35, 38]


def _in_band(age, lo, hi):
    return age is not None and lo <= age <= hi


def build_controls():
    controls = []

    # ── Person: sex by age (B01001). Universe: everyone, GQ included. ────────
    for band in AGE_BANDS:
        lo, hi = band
        for sex_code, sex_name, offset in ((1, "male", 0), (2, "female", 24)):
            varnums = [v + offset for v in B01001_MALE[band]]
            controls.append((
                f"age_{lo}_{hi}_{sex_name}", "person", "b01001", varnums,
                (lambda p, lo=lo, hi=hi, s=sex_code:
                    p["sex"] == s and _in_band(p["agep"], lo, hi)),
            ))

    # ── Person: employment status (B23025). Universe: population 16+. ────────
    # ESR 1/2 employed civilian, 4/5 armed forces (ACS counts them as employed,
    # in variable 006), 3 unemployed, 6 not in the labour force.
    controls += [
        ("emp_employed", "person", "b23025", [4, 6],
         lambda p: p["agep"] >= 16 and p["esr"] in (1, 2, 4, 5)),
        ("emp_unemployed", "person", "b23025", [5],
         lambda p: p["agep"] >= 16 and p["esr"] == 3),
        ("emp_nilf", "person", "b23025", [7],
         lambda p: p["agep"] >= 16 and p["esr"] == 6),
    ]

    # ── Person: disability (B18101). ─────────────────────────────────────────
    # Universe is the CIVILIAN NONINSTITUTIONALISED population: institutional
    # group quarters (typehugq 2) and active-duty military (ESR 4/5) are out.
    def civ_noninst(p):
        return p["typehugq"] != 2 and p["esr"] not in (4, 5)

    controls += [
        ("dis_yes", "person", "b18101", B18101_WITH,
         lambda p: civ_noninst(p) and p["dis"] == 1),
        ("dis_no", "person", "b18101", [v + 1 for v in B18101_WITH],
         lambda p: civ_noninst(p) and p["dis"] == 2),
    ]

    # ── Person: commute mode (B08301). Universe: workers 16+ ─────────────────
    # with a reported means of transportation. JWTRNS: 1 car/truck/van,
    # 2 bus, 3 subway, 4 long-distance rail, 5 light rail, 6 ferry, 7 taxi,
    # 8 motorcycle, 9 bicycle, 10 walked, 11 worked from home, 12 other.
    MODE = {
        "mode_car": ([2], (1,)),
        "mode_transit": ([10], (2, 3, 4, 5, 6)),
        "mode_bike": ([18], (9,)),
        "mode_walk": ([19], (10,)),
        "mode_wfh": ([21], (11,)),
        "mode_other": ([16, 17, 20], (7, 8, 12)),
    }
    for name, (varnums, codes) in MODE.items():
        controls.append((
            name, "person", "b08301", varnums,
            lambda p, codes=codes: p["agep"] >= 16 and p["jwtrns"] in codes,
        ))

    # ── Household: income (B19001), 2023 dollars via ADJINC. ─────────────────
    INCOME = [
        ("inc_lt25k", [2, 3, 4, 5], (None, 25000)),
        ("inc_25_50k", [6, 7, 8, 9, 10], (25000, 50000)),
        ("inc_50_75k", [11, 12], (50000, 75000)),
        ("inc_75_100k", [13], (75000, 100000)),
        ("inc_100_150k", [14, 15], (100000, 150000)),
        ("inc_150_200k", [16], (150000, 200000)),
        ("inc_200k_plus", [17], (200000, None)),
    ]
    for name, varnums, (lo, hi) in INCOME:
        controls.append((
            name, "hh", "b19001", varnums,
            lambda h, lo=lo, hi=hi: (
                h["hincp_adj"] is not None
                and (lo is None or h["hincp_adj"] >= lo)
                and (hi is None or h["hincp_adj"] < hi)),
        ))

    # ── Household: vehicles available (B25044), owner and renter summed. ─────
    VEH = [
        ("veh_0", [3, 10], (0, 0)),
        ("veh_1", [4, 11], (1, 1)),
        ("veh_2", [5, 12], (2, 2)),
        ("veh_3plus", [6, 7, 8, 13, 14, 15], (3, 99)),
    ]
    for name, varnums, (lo, hi) in VEH:
        controls.append((
            name, "hh", "b25044", varnums,
            lambda h, lo=lo, hi=hi: h["veh"] is not None and lo <= h["veh"] <= hi,
        ))

    # ── Household: size (B11016), family and nonfamily summed. ───────────────
    SIZE = [
        ("size_1", [10], (1, 1)),
        ("size_2", [3, 11], (2, 2)),
        ("size_3", [4, 12], (3, 3)),
        ("size_4plus", [5, 6, 7, 8, 13, 14, 15, 16], (4, 99)),
    ]
    for name, varnums, (lo, hi) in SIZE:
        controls.append((
            name, "hh", "b11016", varnums,
            lambda h, lo=lo, hi=hi: lo <= h["np"] <= hi,
        ))

    # ── Household: total occupied units. Anchors the overall scale. ──────────
    controls.append(("hh_total", "hh", "b19001", [1], lambda h: True))

    return controls


CONTROLS = build_controls()
HH_COLS = "serialno, typehugq, np, wgtp, hincp_adj, veh"
P_COLS = "serialno, agep, sex, esr, dis, jwtrns, pwgtp"


def _fetch_puma(cur, state, puma):
    """PUMS units for one PUMA, with their control incidence.

    A "unit" is an occupied housing unit, or a single group-quarters person.
    Vacant units are dropped: they house nobody, so they can supply no agent.
    GQ units carry the person's own weight and contribute to person controls
    only — their household record has no weight and no income by construction.
    """
    cur.execute(f"SELECT {HH_COLS} FROM census.pums_households "
                "WHERE state = %s AND puma = %s", (state, puma))
    hh = {}
    for serialno, typehugq, np_, wgtp, hincp_adj, veh in cur.fetchall():
        if typehugq == 1 and np_ == 0:
            continue                       # vacant
        hh[serialno] = {"serialno": serialno, "typehugq": typehugq, "np": np_,
                        "wgtp": wgtp, "hincp_adj": hincp_adj, "veh": veh,
                        "persons": []}

    cur.execute(f"SELECT {P_COLS} FROM census.pums_persons "
                "WHERE state = %s AND puma = %s", (state, puma))
    for serialno, agep, sex, esr, dis, jwtrns, pwgtp in cur.fetchall():
        unit = hh.get(serialno)
        if unit is None:
            continue
        unit["persons"].append({"agep": agep, "sex": sex, "esr": esr, "dis": dis,
                                "jwtrns": jwtrns, "pwgtp": pwgtp,
                                "typehugq": unit["typehugq"]})

    serials, w0, rows = [], [], []
    for serialno, unit in hh.items():
        if not unit["persons"]:
            continue
        is_gq = unit["typehugq"] != 1
        base = (max(p["pwgtp"] for p in unit["persons"]) if is_gq else unit["wgtp"])
        if base <= 0:
            continue
        inc = np.zeros(len(CONTROLS), dtype=np.float64)
        for j, (_name, level, _tbl, _vars, test) in enumerate(CONTROLS):
            if level == "hh":
                inc[j] = 0.0 if is_gq else float(bool(test(unit)))
            else:
                inc[j] = float(sum(1 for p in unit["persons"] if test(p)))
        serials.append(serialno)
        w0.append(float(base))
        rows.append(inc)

    if not serials:
        return [], np.zeros(0), np.zeros((0, len(CONTROLS)))
    return serials, np.asarray(w0), np.vstack(rows)


def ipu(A, w0, targets, max_iter=MAX_ITER, tol=TOL):
    """Scale weights until every control is met, or the sample runs out of room.

    One sweep visits each control in turn and multiplies the weights that
    contribute to it by (target / achieved), so that control is exactly met;
    the next control disturbs it again, and repeated sweeps converge on weights
    that satisfy all of them. This is IPU: the unit being scaled is a whole
    household, which is why one weight can serve household and person controls
    at once.

    Two exits besides the tolerance: a sweep that no longer moves any weight
    has converged as far as this sample allows, and a hard iteration cap stops
    a tract whose controls the sample simply cannot reconcile. Both are
    reported, never silently accepted.

    Returns (weights, iterations, max floored relative error).
    """
    w = w0.copy()
    active = [j for j in range(A.shape[1]) if targets[j] is not None]
    cols = {j: (A[:, j], A[:, j] > 0) for j in active}
    best_w, best_err = w.copy(), float("inf")
    err = float("inf")
    it = 0
    for it in range(1, max_iter + 1):
        before = w.copy()
        for j in active:
            col, mask = cols[j]
            s = float(col @ w)
            if s <= 0:
                continue                   # no unit can supply this control
            target = targets[j]
            # A control of zero drives its contributors to ~0 rather than
            # exactly 0: an exact zero would remove those households from every
            # other control too, and a later division by their vanished total
            # would end the fit rather than continue it.
            factor = (target if target > 0 else ZERO_TARGET) / s
            w[mask] *= factor

        err = max((abs(float(cols[j][0] @ w) - targets[j])
                   / max(targets[j], ERROR_FLOOR)) for j in active) if active else 0.0
        if err < best_err:
            best_err, best_w = err, w.copy()
        if err <= tol:
            break
        moved = float(np.abs(w - before).max())
        if moved <= WEIGHT_EPS * max(1.0, float(np.abs(w).max())):
            break
    return best_w, it, best_err


def ipu_batch(A, w0, targets_by_tract, max_iter=MAX_ITER, tol=TOL):
    """Fit every tract of one PUMA at once.

    All tracts in a PUMA share the same sample and the same incidence matrix
    and differ only in their targets, so their weight vectors can sit in one
    (units x tracts) matrix and every sweep becomes one numpy operation per
    control instead of one per control per tract.

    Each sweep is arithmetically identical to :func:`ipu` on that column: with
    the stopping rules held the same, the two agree to 3e-13. What differs is
    when they stop — the batch keeps sweeping while any tract is still out of
    tolerance, so a tract that converged early keeps its best sweep and can end
    marginally better than the scalar version would leave it. ~6x faster, which
    is the difference between an hour and ten minutes across 28,000 tracts.

    ``targets_by_tract`` is a list of per-control target lists (None for a
    control this tract has no published figure for). Returns
    ``(W, iterations, errors)``.
    """
    n_units, n_controls = A.shape
    n_tracts = len(targets_by_tract)
    W = np.repeat(w0[:, None], n_tracts, axis=1)

    # A control is fitted only for the tracts that publish it; the rest keep
    # their weights untouched at that step.
    targets = np.zeros((n_controls, n_tracts))
    has = np.zeros((n_controls, n_tracts), dtype=bool)
    for k, tgt in enumerate(targets_by_tract):
        for j, value in enumerate(tgt):
            if value is not None:
                targets[j, k] = value
                has[j, k] = True

    active = [j for j in range(n_controls) if has[j].any()]
    masks = {j: A[:, j] > 0 for j in active}
    denom = np.maximum(targets, ERROR_FLOOR)

    best_W = W.copy()
    best_err = np.full(n_tracts, np.inf)
    it = 0
    for it in range(1, max_iter + 1):
        for j in active:
            col = A[:, j]
            s = col @ W                                  # achieved, per tract
            factor = np.where(
                has[j] & (s > 0),
                np.where(targets[j] > 0, targets[j], ZERO_TARGET) / np.where(s > 0, s, 1.0),
                1.0)
            W[masks[j], :] *= factor[None, :]

        achieved = np.vstack([A[:, j] @ W for j in active])
        err = np.where(has[active], np.abs(achieved - targets[active]) / denom[active], 0.0)
        err = err.max(axis=0)
        improved = err < best_err
        if improved.any():
            best_err = np.where(improved, err, best_err)
            best_W[:, improved] = W[:, improved]
        if (best_err <= tol).all():
            break
    return best_W, it, best_err


def _targets_for_tract(acs, tract):
    """Collapse the tract's ACS cells into one target per control."""
    out = []
    for _name, _level, table, varnums, _test in CONTROLS:
        cells = acs.get((tract, table))
        if cells is None:
            out.append(None)
            continue
        vals = [cells.get(v) for v in varnums]
        out.append(None if any(v is None for v in vals) else float(sum(vals)))
    return out


def resolve_acs_geoids(cur, tracts):
    """Map each tract to the ACS geography that actually describes it.

    Three cases, in order:

    1. The tract has its own ACS estimates. Almost all of them.
    2. **Connecticut.** In 2022 the state replaced its counties with nine
       planning regions as county-equivalents, so ACS 2023 publishes CT tracts
       under planning-region prefixes (09110-09190) while LODES and TIGER 2020
       blocks still carry the old county FIPS (09001, 09009...). The 6-digit
       tract code survives the change: all 423 CT home tracts here match
       exactly one ACS tract on it, so that is the crosswalk. Without this,
       every Connecticut commuter into New York -- 34,341 jobs -- would be
       unplaceable.
    3. **Tracts revised after the 2020 TIGER vintage** the block table was
       built from, which ACS 2023 no longer publishes: 15 tracts in Suffolk
       County, NY, carrying 0.03% of all jobs. They fall back to their
       COUNTY's controls, scaled by the tract's share of county housing units
       (2020 census counts, which do exist for them). The tract then gets its
       county's composition at its own size -- weaker than a real tract fit,
       and reported as such, but far better than dropping the residents.

    Returns ``{tract: (kind, geoid, scale)}`` with kind 'tract' or 'county'.
    """
    tracts = list(tracts)
    cur.execute("""
        SELECT DISTINCT geoid FROM census.acs_estimates
        WHERE table_id = 'b19001' AND sumlevel = 140 AND geoid = ANY(%s)
    """, (tracts,))
    direct = {r[0] for r in cur.fetchall()}

    unresolved = [t for t in tracts if t not in direct]
    resolved = {t: ("tract", t, 1.0) for t in direct}
    if not unresolved:
        return resolved, 0, 0

    # Case 2: same state, same 6-digit tract code, different county prefix.
    states = sorted({t[:2] for t in unresolved})
    cur.execute("""
        SELECT DISTINCT geoid FROM census.acs_estimates
        WHERE table_id = 'b19001' AND sumlevel = 140 AND left(geoid, 2) = ANY(%s)
    """, (states,))
    by_code = defaultdict(list)
    for (geoid,) in cur.fetchall():
        by_code[(geoid[:2], geoid[5:])].append(geoid)

    still = []
    n_recoded = 0
    for t in unresolved:
        candidates = by_code.get((t[:2], t[5:]), [])
        if len(candidates) == 1:
            resolved[t] = ("tract", candidates[0], 1.0)
            n_recoded += 1
        else:
            still.append(t)

    # Case 3: county controls, scaled by the tract's share of county housing.
    if still:
        cur.execute("""
            SELECT substr(geoid20, 1, 11) AS tract,
                   sum(housing20)::float AS tract_housing,
                   sum(sum(housing20)::float) OVER (PARTITION BY substr(geoid20, 1, 5))
            FROM census.blocks
            WHERE substr(geoid20, 1, 5) = ANY(%s)
            GROUP BY substr(geoid20, 1, 11), substr(geoid20, 1, 5)
        """, (sorted({t[:5] for t in still}),))
        share = {}
        for tract, tract_h, county_h in cur.fetchall():
            share[tract] = (tract_h / county_h) if county_h else 0.0
        for t in still:
            scale = share.get(t, 0.0)
            if scale <= 0:
                raise SystemExit(
                    f"FATAL: tract {t} has neither ACS estimates nor any housing "
                    "units to scale its county's by — cannot fit it")
            resolved[t] = ("county", t[:5], scale)

    return resolved, n_recoded, len(still)


def load_acs_for_tracts(cur, tracts, resolved):
    """{(tract, table): {varnum: estimate}} for the tracts being fitted.

    Keyed by the tract being FITTED, not by the geography the numbers came
    from, so the caller never has to know which of the three cases applied.
    """
    tables = sorted({c[2] for c in CONTROLS})

    want_tract = {geoid: [] for kind, geoid, _ in resolved.values() if kind == "tract"}
    want_county = {geoid: [] for kind, geoid, _ in resolved.values() if kind == "county"}
    for tract, (kind, geoid, scale) in resolved.items():
        (want_tract if kind == "tract" else want_county)[geoid].append((tract, scale))

    acs = defaultdict(dict)
    if want_tract:
        cur.execute("""
            SELECT geoid, table_id, varnum, estimate
            FROM census.acs_estimates
            WHERE sumlevel = 140 AND table_id = ANY(%s) AND geoid = ANY(%s)
        """, (tables, list(want_tract)))
        for geoid, table_id, varnum, est in cur.fetchall():
            for tract, _scale in want_tract[geoid]:
                acs[(tract, table_id)][varnum] = est
    if want_county:
        cur.execute("""
            SELECT geoid, table_id, varnum, estimate
            FROM census.acs_estimates
            WHERE sumlevel = 50 AND table_id = ANY(%s) AND geoid = ANY(%s)
        """, (tables, list(want_county)))
        for geoid, table_id, varnum, est in cur.fetchall():
            for tract, scale in want_county[geoid]:
                acs[(tract, table_id)][varnum] = None if est is None else est * scale
    return acs


def needed_tracts(cur):
    """Every tract an agent home can land in: LODES home tracts, plus the core
    blocks the non-worker draw uses."""
    cur.execute("""
        SELECT DISTINCT substr(h_geoid, 1, 11) FROM public.metro_od_pairs
        UNION
        SELECT DISTINCT tract_geoid FROM public.metro_home_blocks
    """)
    return [r[0] for r in cur.fetchall()]


def phase_fit(conn, args) -> None:
    cur = conn.cursor()
    tracts = needed_tracts(cur)
    if args.limit:
        tracts = tracts[:args.limit]
    print(f"  {len(tracts):,} tracts to fit", flush=True)

    cur.execute("SELECT tract_geoid, state, puma FROM census.tract_puma "
                "WHERE tract_geoid = ANY(%s)", (tracts,))
    puma_of = {t: (s, p) for t, s, p in cur.fetchall()}
    missing = [t for t in tracts if t not in puma_of]
    if missing:
        raise SystemExit(f"FATAL: {len(missing)} tracts have no PUMA "
                         f"(first: {missing[:3]}) — the crosswalk is incomplete")

    by_puma = defaultdict(list)
    for t in tracts:
        by_puma[puma_of[t]].append(t)
    print(f"  spanning {len(by_puma):,} PUMAs", flush=True)

    resolved, n_recoded, n_county = resolve_acs_geoids(cur, tracts)
    if n_recoded or n_county:
        print(f"  ACS geography: {n_recoded:,} tracts matched by tract code "
              f"(Connecticut planning regions), {n_county:,} fell back to "
              f"scaled county controls", flush=True)
    acs = load_acs_for_tracts(cur, tracts, resolved)

    write = conn.cursor()
    t0 = time.time()
    done = fitted = failed = 0
    n_weights = 0
    write.execute("DELETE FROM census.pums_tract_weights WHERE tract_geoid = ANY(%s)",
                  (tracts,))
    write.execute("DELETE FROM census.pums_tract_weight_meta WHERE tract_geoid = ANY(%s)",
                  (tracts,))

    for (state, puma), puma_tracts in by_puma.items():
        serials, w0, A = _fetch_puma(cur, state, puma)
        if not serials:
            raise SystemExit(f"FATAL: PUMA {state}-{puma} has no usable PUMS units")

        targets = []
        for tract in puma_tracts:
            tgt = _targets_for_tract(acs, tract)
            if all(t is None for t in tgt):
                raise SystemExit(f"FATAL: tract {tract} has no ACS controls")
            targets.append(tgt)

        # Every tract in this PUMA is fitted in one pass: they share the sample
        # and the incidence matrix and differ only in their targets.
        W, iters, errs = ipu_batch(A, w0, targets)
        serial_arr = np.asarray(serials)

        # One COPY and one INSERT per PUMA rather than per tract: a PUMA holds
        # ~34 tracts and the whole run ~1,400 PUMAs, so per-tract statements
        # would mean 28,000 round trips to save nothing.
        meta_rows = []
        with write.copy("COPY census.pums_tract_weights "
                        "(tract_geoid, serialno, weight) FROM STDIN") as cp:
            for k, tract in enumerate(puma_tracts):
                w, err = W[:, k], float(errs[k])
                keep = w >= MIN_WEIGHT
                for serialno, weight in zip(serial_arr[keep], w[keep]):
                    cp.write_row((tract, str(serialno), float(weight)))
                n_kept = int(keep.sum())
                n_weights += n_kept
                meta_rows.append((tract, puma, n_kept, iters, err, bool(err <= TOL)))
                done += 1
                fitted += int(err <= TOL)
                failed += int(err > TOL)
        write.executemany(
            "INSERT INTO census.pums_tract_weight_meta "
            "(tract_geoid, puma, n_units, iterations, max_rel_error, converged) "
            "VALUES (%s, %s, %s, %s, %s, %s)", meta_rows)
        if done % 500 < len(puma_tracts):
            conn.commit()
            rate = done / max(time.time() - t0, 1e-9)
            print(f"    {done:,}/{len(tracts):,} tracts  "
                  f"{n_weights:,} weights  {rate:.1f} tracts/s  "
                  f"{failed:,} outside {TOL:.0%}", flush=True)
    conn.commit()
    print(f"  done: {done:,} tracts ({fitted:,} within {TOL:.0%}, {failed:,} not), "
          f"{n_weights:,} weights, {time.time() - t0:.0f}s")
    print("  NOTE: a residual is expected — ACS's own tract tables disagree with "
          "each other (household-size against population totals differ by several "
          "percent), so no weighting can satisfy every control exactly.")


def _person_control_totals(cur, state, puma):
    """Person-control totals for a PUMA using PUMS PERSON weights.

    Validation must weight people by PWGTP, not by their household's WGTP. The
    two are calibrated separately — households to household counts, people to
    population — so summing household weights over people understates the
    population, most for children, whose person weights diverge furthest.
    Measured on California before this was split out: household controls landed
    within 0.1% while every person control sat 2-7% low, which is that artefact
    and not a universe error.

    The fitting still STARTS from WGTP, which is correct: IPU carries one weight
    per household and moves it until the person controls are met as well.
    """
    cur.execute(f"SELECT {P_COLS}, h.typehugq FROM census.pums_persons p "
                "JOIN census.pums_households h USING (serialno) "
                "WHERE p.state = %s AND p.puma = %s", (state, puma))
    totals = np.zeros(len(CONTROLS))
    for serialno, agep, sex, esr, dis, jwtrns, pwgtp, typehugq in cur.fetchall():
        person = {"agep": agep, "sex": sex, "esr": esr, "dis": dis,
                  "jwtrns": jwtrns, "pwgtp": pwgtp, "typehugq": typehugq}
        for j, (_name, level, _tbl, _vars, test) in enumerate(CONTROLS):
            if level == "person" and test(person):
                totals[j] += pwgtp
    return totals


def phase_validate(conn, args) -> None:
    """Compare each PUMS recode against ACS, summed over a whole state.

    PUMAs nest inside states exactly, so a state total is the one place the two
    sources must agree cell for cell. A gap here means the recode's universe is
    wrong — which would otherwise show up only as a tract that cannot be fitted.

    Household controls are checked with household weights over occupied units;
    person controls with person weights. See _person_control_totals.
    """
    cur = conn.cursor()
    if args.states:
        states = list(args.states)
    else:
        cur.execute("SELECT DISTINCT state FROM census.pums_persons ORDER BY state")
        states = [r[0] for r in cur.fetchall()]
    print(f"  validating {len(CONTROLS)} controls over {len(states)} states\n")
    print(f"    {'control':22s} {'PUMS':>13s} {'ACS':>13s} {'diff':>9s}")

    worst = []
    for state in states:
        cur.execute("SELECT DISTINCT puma FROM census.pums_persons WHERE state = %s",
                    (state,))
        pumas = [r[0] for r in cur.fetchall()]
        is_person = np.array([c[1] == "person" for c in CONTROLS])
        pums_tot = np.zeros(len(CONTROLS))
        for puma in pumas:
            serials, w0, A = _fetch_puma(cur, state, puma)
            if serials:
                pums_tot += np.where(is_person, 0.0, A.T @ w0)   # household weights
            pums_tot += np.where(is_person, _person_control_totals(cur, state, puma), 0.0)

        cur.execute("""
            SELECT table_id, varnum, sum(estimate)
            FROM census.acs_estimates
            WHERE sumlevel = 50 AND left(geoid, 2) = %s
            GROUP BY table_id, varnum
        """, (state,))
        acs_state = defaultdict(dict)
        for table_id, varnum, total in cur.fetchall():
            acs_state[table_id][varnum] = float(total or 0.0)

        print(f"  state {state}")
        for j, (name, _level, table, varnums, _test) in enumerate(CONTROLS):
            acs_val = sum(acs_state.get(table, {}).get(v, 0.0) for v in varnums)
            pums_val = pums_tot[j]
            diff = (pums_val - acs_val) / acs_val * 100 if acs_val else float("nan")
            flag = "  <-- CHECK" if acs_val and abs(diff) > 2.0 else ""
            print(f"    {name:22s} {pums_val:13,.0f} {acs_val:13,.0f} "
                  f"{diff:8.2f}%{flag}")
            if acs_val:
                worst.append((abs(diff), name, state, diff))

    worst.sort(reverse=True)
    print("\n  largest gaps:")
    for _, name, state, diff in worst[:8]:
        print(f"    {name:22s} state {state}  {diff:+.2f}%")


def phase_home_blocks(conn, args) -> None:
    """Populate public.metro_home_blocks: populated CORE blocks per metro."""
    cur = conn.cursor()
    cur.execute("TRUNCATE public.metro_home_blocks")
    cur.execute("""
        INSERT INTO public.metro_home_blocks
            (metro, geoid, tract_geoid, pop20, lat, lon)
        SELECT cb.metro, b.geoid20, substr(b.geoid20, 1, 11), b.pop20,
               ST_Y(b.pt), ST_X(b.pt)
        FROM census.core_blocks cb
        JOIN census.blocks b ON b.geoid20 = cb.geoid20
        WHERE b.pop20 > 0
    """)
    conn.commit()
    for metro, blocks, pop in cur.execute("""
        SELECT metro, count(*), sum(pop20) FROM public.metro_home_blocks
        GROUP BY metro ORDER BY metro
    """).fetchall():
        print(f"    {metro:8s} {blocks:7,d} populated core blocks  {pop:10,d} residents")


PHASES = {"home-blocks": phase_home_blocks, "validate": phase_validate, "fit": phase_fit}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--what", choices=list(PHASES), required=True)
    ap.add_argument("--limit", type=int, default=0, help="fit only the first N tracts")
    ap.add_argument("--states", nargs="*", default=None,
                    help="validate only these state FIPS (default: all loaded)")
    args = ap.parse_args()
    with psycopg.connect(DSN) as conn:
        conn.cursor().execute("SET work_mem = '1GB'")
        print(f"phase: {args.what}", flush=True)
        PHASES[args.what](conn, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())

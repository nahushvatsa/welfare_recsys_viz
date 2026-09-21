#!/usr/bin/env python3
"""Load ACS 5-year detailed tables into census.acs_estimates.

    python3 db/load_acs.py                 # every table in ACS_TABLES
    python3 db/load_acs.py --tables b19001 b25044

Input: the pipe-delimited ``.dat`` files fetched by db/fetch_census_data.py.
Each holds one table for EVERY geography in the country, with a header of
``GEO_ID|B19001_E001|B19001_M001|...`` — estimate and margin of error side by
side. Only the geographies this project can use are kept:

* summary level 140 (tract) and 150 (block group), the two the reweighting and
  its diagnostics read, plus
* summary level 50 (county), kept purely so a load can be sanity-checked
  against a published county total without another download,

and only in the seven states behind the eight metros. That cuts ~330k
geographies per table to ~121k.

Idempotent per table: the table's rows are deleted and reloaded.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time

import psycopg

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DSN = os.environ.get("WELFARE_LOADER_DSN", "")
DATA = os.path.join(REPO, "data", "census")
ACS_DIR = os.path.join(DATA, "acs")
MANIFEST = os.path.join(DATA, "manifest_acs_pums.json")

YEAR = 2023

# Every state an agent home can fall in: the seven holding the metro cores,
# plus the seven their commuter sheds reach (NJ, MD, VA, CT, IN, PA, DE). The
# controls are needed wherever an agent LIVES, so this must match the PUMS
# state list in db/load_pums.py — a tract with no controls cannot be fitted,
# and db/build_tract_weights.py stops rather than fitting it to nothing.
STATES = ("36", "06", "17", "48", "12", "53", "11",
          "34", "24", "51", "09", "18", "42", "10")

# Summary levels kept, by GEO_ID prefix.
SUMLEVELS = {"1400000US": 140, "1500000US": 150, "0500000US": 50}

# table -> what it controls. Order is the load order.
ACS_TABLES = {
    "b01001": "sex by age — person control",
    "b19001": "household income in the past 12 months — household control",
    "b23025": "employment status for the population 16+ — person control",
    "b25044": "tenure by vehicles available — household control",
    "b08301": "means of transportation to work — person control",
    "b11016": "household type by household size — household control",
    "b18101": "sex by age by disability status — person control (tract only)",
    "b08201": "household size by vehicles available (tract only)",
}

# ACS jam values: suppressed / not-applicable cells are published as large
# negative sentinels, never as real counts. Storing them as numbers would put
# -999,999,999 households in a control total.
JAM_FLOOR = -111111111


def verify_checksum(path: str) -> None:
    files = json.load(open(MANIFEST))["files"]
    entry = files.get(os.path.basename(path))
    if entry is None:
        raise SystemExit(f"FATAL: {os.path.basename(path)} is not in the manifest")
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    if h.hexdigest() != entry["sha256"]:
        raise SystemExit(f"FATAL: {os.path.basename(path)} does not match its manifest digest")


def _parse(token: str):
    """Return a float, or None for blank and for ACS jam values."""
    token = token.strip()
    if token in ("", ".", "null"):
        return None
    try:
        val = float(token)
    except ValueError:
        return None
    return None if val <= JAM_FLOOR else val


def load_table(conn, table_id: str, log=print) -> int:
    path = os.path.join(ACS_DIR, f"acsdt5y{YEAR}-{table_id}.dat")
    if not os.path.exists(path):
        raise SystemExit(f"FATAL: {path} not found — run db/fetch_census_data.py --what acs")
    verify_checksum(path)

    cur = conn.cursor()
    cur.execute("DELETE FROM census.acs_estimates WHERE table_id = %s", (table_id,))

    t0 = time.time()
    kept = skipped = suppressed = 0
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f, delimiter="|")
        header = next(reader)
        if header[0] != "GEO_ID":
            raise SystemExit(f"FATAL: {path} header starts with {header[0]!r}, expected GEO_ID")

        # Pair each estimate column with its margin-of-error column by variable
        # number, rather than assuming they alternate.
        est_col, moe_col = {}, {}
        for i, name in enumerate(header[1:], start=1):
            body = name.split("_")[-1]          # 'E001' / 'M001'
            kind, num = body[0], int(body[1:])
            (est_col if kind == "E" else moe_col)[num] = i
        variables = sorted(est_col)
        if not variables:
            raise SystemExit(f"FATAL: no estimate columns parsed from {path}")

        with cur.copy("COPY census.acs_estimates "
                      "(geoid, sumlevel, table_id, varnum, estimate, moe) FROM STDIN") as cp:
            for row in reader:
                geo = row[0]
                prefix = geo[:9]
                sumlevel = SUMLEVELS.get(prefix)
                if sumlevel is None:
                    skipped += 1
                    continue
                geoid = geo[9:]
                if not geoid.startswith(STATES):
                    skipped += 1
                    continue
                for num in variables:
                    est = _parse(row[est_col[num]])
                    moe = _parse(row[moe_col[num]]) if num in moe_col else None
                    if est is None:
                        suppressed += 1
                    cp.write_row((geoid, sumlevel, table_id, num, est, moe))
                    kept += 1
    conn.commit()
    log(f"  {table_id}: {kept:,} cells kept ({len(variables)} variables), "
        f"{skipped:,} geographies skipped, {suppressed:,} suppressed cells, "
        f"{time.time() - t0:.0f}s")
    return kept


def verify(conn, log=print) -> None:
    cur = conn.cursor()
    log("\n  loaded, by table and summary level:")
    for table_id, sumlevel, geos, cells in cur.execute("""
        SELECT table_id, sumlevel, count(DISTINCT geoid), count(*)
        FROM census.acs_estimates GROUP BY 1, 2 ORDER BY 1, 2
    """).fetchall():
        log(f"    {table_id}  sumlevel {sumlevel:3d}  {geos:>7,} geographies  {cells:>9,} cells")

    # A tract's households must equal the sum of its income buckets: B19001_001
    # is the universe and 002..017 partition it. If that fails, estimate and
    # margin columns were paired wrong.
    bad = cur.execute("""
        WITH t AS (
            SELECT geoid,
                   max(estimate) FILTER (WHERE varnum = 1) AS total,
                   sum(estimate) FILTER (WHERE varnum BETWEEN 2 AND 17) AS parts
            FROM census.acs_estimates
            WHERE table_id = 'b19001' AND sumlevel = 140
            GROUP BY geoid
        )
        SELECT count(*) FROM t WHERE total IS DISTINCT FROM parts
    """).fetchone()[0]
    log(f"  b19001 tracts where total <> sum of buckets: {bad:,}")
    if bad:
        raise SystemExit("FATAL: income buckets do not sum to the table total — "
                         "estimate/MOE column pairing is wrong")

    # Home tracts whose own ACS estimates are missing. This is reported, not
    # fatal: two known geography mismatches are resolved downstream by
    # db/build_tract_weights.py (resolve_acs_geoids) — Connecticut's 2022
    # switch to planning regions, which renumbers tract prefixes, and a handful
    # of tracts revised after the 2020 TIGER vintage. What WOULD be fatal is a
    # tract that cannot be resolved by either route, and that check belongs
    # where the resolution happens.
    rows = cur.execute("""
        SELECT left(t.tract, 2) AS state, count(*)
        FROM (SELECT DISTINCT substr(h_geoid, 1, 11) AS tract
              FROM public.metro_od_pairs) t
        LEFT JOIN (SELECT DISTINCT geoid FROM census.acs_estimates
                   WHERE table_id = 'b19001' AND sumlevel = 140) a
          ON a.geoid = t.tract
        WHERE a.geoid IS NULL
        GROUP BY 1 ORDER BY 2 DESC
    """).fetchall()
    total = sum(n for _s, n in rows)
    log(f"  home tracts without their own ACS controls: {total:,}"
        + (" (" + ", ".join(f"state {st}: {n:,}" for st, n in rows) + ")" if rows else ""))
    if total:
        log("    -> resolved by db/build_tract_weights.py; it fails loudly on any "
            "tract it cannot resolve")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tables", nargs="*", default=list(ACS_TABLES))
    ap.add_argument("--skip-verify", action="store_true")
    args = ap.parse_args()

    unknown = [t for t in args.tables if t not in ACS_TABLES]
    if unknown:
        raise SystemExit(f"FATAL: unknown tables {unknown}; known: {list(ACS_TABLES)}")

    with psycopg.connect(DSN) as conn:
        conn.cursor().execute("SET maintenance_work_mem = '2GB'")
        for table_id in args.tables:
            print(f"loading {table_id} — {ACS_TABLES[table_id]}", flush=True)
            load_table(conn, table_id)
        if not args.skip_verify:
            verify(conn)
    return 0


if __name__ == "__main__":
    sys.exit(main())

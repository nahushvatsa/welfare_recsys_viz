#!/usr/bin/env python3
"""Export built populations to CSV, for running the engine without Postgres.

    python3 db/export_population.py --metros dc --replicates 0 1
    python3 db/export_population.py --all

Writes into ``data/population`` (gitignored — populations are data):

    <metro>_r<NN>.csv          one (metro, replicate) slice, in agent_id order
    <metro>_home_blocks.csv    core blocks, for the routing endpoint universe

``welfare_rs.datasource.LocalDataSource`` reads exactly these files, and both
paths run every row through ``coerce_population_row``, so a CSV-backed run and
a database-backed run build identical agents.

This is also how the demo population is produced: a small metro exported once
lets someone clone the repo and run a simulation without standing up a
database — see README.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import psycopg

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DSN = os.environ.get("WELFARE_LOADER_DSN", "")
OUT_DIR = os.path.join(REPO, "data", "population")


def export_population(cur, metro: str, replicate: int, log=print) -> int:
    cur.execute("""
        SELECT * FROM public.metro_population
        WHERE metro = %s AND replicate = %s ORDER BY agent_id
    """, (metro, replicate))
    cols = [d.name for d in cur.description]
    rows = cur.fetchall()
    if not rows:
        log(f"    {metro} r{replicate:02d}: nothing built, skipped")
        return 0
    path = os.path.join(OUT_DIR, f"{metro}_r{replicate:02d}.csv")
    tmp = path + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        w.writerows(rows)
    os.replace(tmp, path)
    log(f"    {metro} r{replicate:02d}: {len(rows):,} agents -> {os.path.basename(path)}")
    return len(rows)


def export_home_blocks(cur, metro: str, log=print) -> int:
    cur.execute("""
        SELECT geoid, tract_geoid, pop20, lat, lon
        FROM public.metro_home_blocks WHERE metro = %s ORDER BY geoid
    """, (metro,))
    rows = cur.fetchall()
    if not rows:
        return 0
    path = os.path.join(OUT_DIR, f"{metro}_home_blocks.csv")
    tmp = path + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["geoid", "tract_geoid", "pop20", "lat", "lon"])
        w.writerows(rows)
    os.replace(tmp, path)
    log(f"    {metro}: {len(rows):,} core home blocks -> {os.path.basename(path)}")
    return len(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metros", nargs="*", default=None)
    ap.add_argument("--replicates", nargs="*", type=int, default=None)
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    with psycopg.connect(DSN) as conn:
        cur = conn.cursor()
        if args.all or not args.metros:
            cur.execute("SELECT DISTINCT metro FROM public.metro_population ORDER BY metro")
            metros = [r[0] for r in cur.fetchall()]
        else:
            metros = args.metros
        if not metros:
            raise SystemExit("FATAL: no populations built — run db/build_population.py first")

        total = 0
        for metro in metros:
            if args.replicates is None:
                cur.execute("SELECT DISTINCT replicate FROM public.metro_population "
                            "WHERE metro = %s ORDER BY replicate", (metro,))
                reps = [r[0] for r in cur.fetchall()]
            else:
                reps = args.replicates
            for rep in reps:
                total += export_population(cur, metro, rep)
            export_home_blocks(cur, metro)
        print(f"done: {total:,} agent rows -> {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

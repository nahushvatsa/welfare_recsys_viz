#!/usr/bin/env python3
"""Download LODES8 origin-destination files and load them into census.lodes_od.

    python3 db/load_lodes_od.py                      # all 7 metro states
    python3 db/load_lodes_od.py --states ny dc
    python3 db/load_lodes_od.py --year 2023 --jt JT01

BOTH PARTS ARE LOADED, and that is not optional:
    main : worker lives in the same state as the job
    aux  : worker lives OUT of state
On dc_od_JT01_2023 main carries 203,088 jobs and aux 422,114 -- two thirds of
DC's workforce commutes in from Maryland and Virginia and exists only in aux.

WORK-COUNTY PREFILTER. A job can only land in a metro core if its work block
sits in a county that intersects that core polygon, so rows are filtered on
left(w_geocode, 5) during the stream. This is an exact SUPERSET of the final
selection -- the authoritative test is still ST_Contains against
geo.metro_cores in db/derive_metro_flows.py -- but it cuts a statewide file
(all of New York) down to the handful of counties the core actually touches.

Idempotent per (state, part, year, jt): those rows are deleted before reload.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import os
import sys
import time

import psycopg

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DSN = os.environ.get("WELFARE_LOADER_DSN", "")  # "" -> libpq env vars
DL_DIR = os.path.join(REPO, "data", "census", "lodes")

BASE = "https://lehd.ces.census.gov/data/lodes/LODES8"

# The 7 states holding the 8 metro cores. Home states are discovered later,
# from the OD itself, and only their TIGER blocks are needed (not their OD).
DEFAULT_STATES = ["ny", "wa", "ca", "il", "tx", "dc", "fl"]

# Source column order in every LODES OD file.
SRC = ["w_geocode", "h_geocode", "S000",
       "SA01", "SA02", "SA03", "SE01", "SE02", "SE03",
       "SI01", "SI02", "SI03", "createdate"]
DST = ["w_geocode", "h_geocode", "part", "st", "year", "jt", "s000",
       "sa01", "sa02", "sa03", "se01", "se02", "se03",
       "si01", "si02", "si03"]


def _download(url: str, path: str, redownload: bool = False) -> str:
    import requests

    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path) and os.path.getsize(path) > 0 and not redownload:
        print(f"    cached  {os.path.basename(path)} "
              f"({os.path.getsize(path) / 1e6:.1f} MB)", flush=True)
        return path
    tmp = path + ".part"
    t0 = time.time()
    with requests.get(url, stream=True, timeout=180) as r:
        r.raise_for_status()
        done = 0
        with open(tmp, "wb") as f:
            for block in r.iter_content(chunk_size=1 << 20):
                f.write(block)
                done += len(block)
    os.replace(tmp, path)
    print(f"    fetched {os.path.basename(path)} ({done / 1e6:.1f} MB, "
          f"{time.time() - t0:.0f}s)", flush=True)
    return path


def _work_counties(conn) -> dict:
    """{state_fips: {county_geoid, ...}} for counties intersecting any core.

    Slivers are kept deliberately: a boundary polygon can clip a neighbouring
    county by a few hundred m^2, and it is the exact ST_Contains test -- not
    this prefilter -- that decides membership.
    """
    cur = conn.cursor()
    cur.execute("""
        SELECT DISTINCT c.statefp, c.geoid
        FROM geo.metro_cores m
        JOIN census.counties c ON ST_Intersects(m.geom, c.geom)
        WHERE ST_Area(ST_Intersection(m.geom, c.geom)) > 0
    """)
    out: dict = {}
    for statefp, geoid in cur.fetchall():
        out.setdefault(statefp, set()).add(geoid)
    return out


def _usps_to_fips(conn) -> dict:
    cur = conn.cursor()
    cur.execute("SELECT lower(usps), statefp FROM census.states")
    return dict(cur.fetchall())


def load_one(conn, st: str, part: str, year: int, jt: str,
             allowed: set, redownload: bool) -> tuple:
    """Stream one OD file into census.lodes_od. Returns (kept, scanned, jobs)."""
    fname = f"{st}_od_{part}_{jt}_{year}.csv.gz"
    url = f"{BASE}/{st}/od/{fname}"
    path = _download(url, os.path.join(DL_DIR, fname), redownload)

    cur = conn.cursor()
    cur.execute(
        "DELETE FROM census.lodes_od WHERE st=%s AND part=%s AND year=%s AND jt=%s",
        (st, part, year, jt),
    )

    kept = scanned = jobs = 0
    t0 = time.time()
    copy_sql = f"COPY census.lodes_od ({', '.join(DST)}) FROM STDIN"
    with gzip.open(path, "rt", newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        # Rows are read POSITIONALLY below, so a column reorder upstream would
        # silently load earnings into the age columns. Assert the full header.
        if header != SRC:
            raise SystemExit(
                f"{fname}: unexpected column layout.\n  expected {SRC}\n  got      {header}")
        with cur.copy(copy_sql) as copy:
            for row in reader:
                scanned += 1
                w = row[0]
                if w[:5] not in allowed:
                    continue
                jobs += int(row[2])
                copy.write_row((
                    w, row[1], part, st, year, jt,
                    int(row[2]),
                    int(row[3]), int(row[4]), int(row[5]),
                    int(row[6]), int(row[7]), int(row[8]),
                    int(row[9]), int(row[10]), int(row[11]),
                ))
                kept += 1
    print(f"    {part}: kept {kept:,} of {scanned:,} rows "
          f"({jobs:,} jobs) in {time.time() - t0:.0f}s", flush=True)
    return kept, scanned, jobs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--states", nargs="*", default=None)
    ap.add_argument("--year", type=int, default=int(os.environ.get("LODES_YEAR", 2023)))
    ap.add_argument("--jt", default=os.environ.get("LODES_JT", "JT01"))
    ap.add_argument("--redownload", action="store_true")
    args = ap.parse_args()

    states = [s.lower() for s in (args.states or DEFAULT_STATES)]
    print(f"LODES8 {args.jt} {args.year} -> census.lodes_od", flush=True)

    total_kept = total_scanned = total_jobs = 0
    with psycopg.connect(DSN) as conn:
        work_counties = _work_counties(conn)
        fips = _usps_to_fips(conn)

        for st in states:
            sf = fips.get(st)
            if sf is None:
                raise SystemExit(f"Unknown state code {st!r}")
            allowed = work_counties.get(sf, set())
            if not allowed:
                print(f"  {st}: no core-intersecting counties — skipping")
                continue
            print(f"  {st} (work counties: {', '.join(sorted(allowed))})", flush=True)
            for part in ("main", "aux"):
                k, s, j = load_one(conn, st, part, args.year, args.jt,
                                   allowed, args.redownload)
                total_kept += k
                total_scanned += s
                total_jobs += j
            conn.commit()

    print(f"\nOK: {total_kept:,} rows kept of {total_scanned:,} scanned "
          f"({total_jobs:,} jobs) in census.lodes_od")
    return 0


if __name__ == "__main__":
    sys.exit(main())

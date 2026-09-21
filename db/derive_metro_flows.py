#!/usr/bin/env python3
"""Derive the commute serving tables from staged LODES + TIGER.

Run the phases in order; each is idempotent and safe to repeat.

    python3 db/derive_metro_flows.py core-blocks    # blocks inside each core
    python3 db/derive_metro_flows.py od-core        # OD rows working in a core
    python3 db/derive_metro_flows.py home-states    # which TIGER states to fetch
    python3 db/derive_metro_flows.py flows          # -> public.metro_county_flows
    python3 db/derive_metro_flows.py pairs          # -> public.metro_od_pairs
    python3 db/derive_metro_flows.py boundaries     # county polygons -> disk cache

`flows` deliberately does NOT need home-state blocks: a home county is the
first 5 characters of the home block GEOID, and county names come from the
national TIGER county file. Only `pairs` needs home-block coordinates, which is
why `home-states` sits between them.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time

import psycopg

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DSN = os.environ.get("WELFARE_LOADER_DSN", "")  # "" -> libpq env vars

LODES_YEAR = int(os.environ.get("LODES_YEAR", 2023))
LODES_JT = os.environ.get("LODES_JT", "JT01")


def _params():
    """Load params.py standalone (welfare_rs/__init__ imports the geo stack)."""
    spec = importlib.util.spec_from_file_location(
        "wrs_params", os.path.join(REPO, "welfare_rs", "params.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _metros(conn, requested):
    cur = conn.cursor()
    cur.execute("SELECT metro FROM geo.metro_cores ORDER BY metro")
    known = [r[0] for r in cur.fetchall()]
    if not requested:
        return known
    unknown = [m for m in requested if m not in known]
    if unknown:
        raise SystemExit(f"Unknown metro(s) {unknown}; known: {known}")
    return list(requested)


# ── Phase: core-blocks ───────────────────────────────────────────────────────

def phase_core_blocks(conn, metros) -> None:
    cur = conn.cursor()
    for metro in metros:
        t0 = time.time()
        cur.execute("DELETE FROM census.core_blocks WHERE metro = %s", (metro,))
        cur.execute("""
            INSERT INTO census.core_blocks (metro, geoid20)
            SELECT m.metro, b.geoid20
            FROM geo.metro_cores m
            JOIN census.blocks b ON ST_Contains(m.geom, b.pt)
            WHERE m.metro = %s
        """, (metro,))
        n = cur.rowcount
        cur.execute(
            "SELECT count(DISTINCT left(geoid20,5)) FROM census.core_blocks "
            "WHERE metro = %s", (metro,))
        counties = cur.fetchone()[0]
        conn.commit()
        print(f"  {metro:8s} {n:8,d} blocks in core "
              f"({counties} counties)  {time.time() - t0:5.1f}s")


# ── Phase: od-core ───────────────────────────────────────────────────────────

def phase_od_core(conn, metros) -> None:
    cur = conn.cursor()
    for metro in metros:
        t0 = time.time()
        cur.execute("DELETE FROM census.od_core WHERE metro = %s", (metro,))
        cur.execute("""
            INSERT INTO census.od_core
                (metro, w_geocode, h_geocode, jobs,
                 sa01, sa02, sa03, se01, se02, se03, si01, si02, si03)
            SELECT cb.metro, o.w_geocode, o.h_geocode, o.s000,
                   o.sa01, o.sa02, o.sa03, o.se01, o.se02, o.se03,
                   o.si01, o.si02, o.si03
            FROM census.lodes_od o
            JOIN census.core_blocks cb
              ON cb.geoid20 = o.w_geocode AND cb.metro = %s
            WHERE o.year = %s AND o.jt = %s
        """, (metro, LODES_YEAR, LODES_JT))
        n = cur.rowcount
        cur.execute("SELECT coalesce(sum(jobs),0), count(DISTINCT w_geocode), "
                    "count(DISTINCT h_geocode) FROM census.od_core WHERE metro=%s",
                    (metro,))
        jobs, wblocks, hblocks = cur.fetchone()
        conn.commit()
        print(f"  {metro:8s} {n:9,d} pairs  {jobs:9,d} jobs  "
              f"{wblocks:7,d} work blocks  {hblocks:8,d} home blocks  "
              f"{time.time() - t0:5.1f}s")


# ── Phase: home-states ───────────────────────────────────────────────────────

def phase_home_states(conn, metros) -> None:
    cur = conn.cursor()
    cur.execute("""
        SELECT left(o.h_geocode, 2) AS statefp,
               s.usps,
               sum(o.jobs) AS jobs,
               count(*) FILTER (WHERE b.geoid20 IS NULL) AS rows_missing_block
        FROM census.od_core o
        LEFT JOIN census.states s ON s.statefp = left(o.h_geocode, 2)
        LEFT JOIN census.blocks b ON b.geoid20 = o.h_geocode
        GROUP BY 1, 2
        ORDER BY jobs DESC
    """)
    rows = cur.fetchall()
    missing = []
    print(f"  {'st':>3} {'usps':>4} {'jobs':>12} {'rows w/o block':>15}")
    for statefp, usps, jobs, miss in rows:
        flag = ""
        if miss:
            missing.append(statefp)
            flag = "  <- TIGER blocks needed"
        print(f"  {statefp:>3} {usps or '??':>4} {jobs:12,d} {miss:15,d}{flag}")
    if missing:
        print("\n  Fetch them with:")
        print(f"    .venv/bin/python db/fetch_tiger.py blocks --states "
              f"{' '.join(sorted(missing))}")
    else:
        print("\n  All home blocks are present.")


# ── Phase: flows ─────────────────────────────────────────────────────────────

FLOWS_SQL = """
INSERT INTO public.metro_county_flows
    (metro, county_fips, county_name, geocode, workers, core_share)
SELECT o.metro,
       left(o.h_geocode, 5) AS county_fips,
       c.namelsad,
       -- Nominatim-style query string. DC's county IS its state, so the plain
       -- "<county>, <state>, USA" form would read "District of Columbia,
       -- District of Columbia, USA".
       CASE WHEN c.namelsad = s.name THEN c.namelsad || ', USA'
            ELSE c.namelsad || ', ' || s.name || ', USA' END,
       sum(o.jobs)::int,
       (coalesce(sum(o.jobs) FILTER (WHERE hb.geoid20 IS NOT NULL), 0)::double precision
        / NULLIF(sum(o.jobs), 0))::real
FROM census.od_core o
JOIN census.counties c ON c.geoid = left(o.h_geocode, 5)
JOIN census.states  s ON s.statefp = c.statefp
LEFT JOIN census.core_blocks hb
       ON hb.geoid20 = o.h_geocode AND hb.metro = o.metro
WHERE o.metro = %s
GROUP BY o.metro, left(o.h_geocode, 5), c.namelsad, s.name
"""


def phase_flows(conn, metros) -> None:
    cur = conn.cursor()
    # A home county missing from TIGER would be silently dropped by the join
    # above, so prove the set is empty before trusting any of it. (Connecticut
    # is the live risk: it replaced counties with planning regions in 2022, so
    # a 2020-vintage block GEOID can carry a county code TIGER2023 no longer
    # lists.)
    cur.execute("""
        SELECT DISTINCT left(o.h_geocode, 5)
        FROM census.od_core o
        LEFT JOIN census.counties c ON c.geoid = left(o.h_geocode, 5)
        WHERE c.geoid IS NULL
        ORDER BY 1
    """)
    orphans = [r[0] for r in cur.fetchall()]
    if orphans:
        raise SystemExit(
            f"{len(orphans)} home county FIPS absent from census.counties: "
            f"{orphans[:20]}\nThese commuters would be dropped silently. "
            "Load a county vintage matching the 2020 block GEOIDs before "
            "deriving flows.")

    for metro in metros:
        t0 = time.time()
        cur.execute("DELETE FROM public.metro_county_flows WHERE metro = %s", (metro,))
        cur.execute(FLOWS_SQL, (metro,))
        n = cur.rowcount
        cur.execute("SELECT sum(workers) FROM public.metro_county_flows WHERE metro=%s",
                    (metro,))
        total = cur.fetchone()[0] or 0
        conn.commit()
        print(f"  {metro:8s} {n:4d} counties  {total:9,d} workers  "
              f"{time.time() - t0:5.1f}s")


# ── Phase: pairs ─────────────────────────────────────────────────────────────

PAIRS_SQL = """
INSERT INTO public.metro_od_pairs
    (metro, h_geoid, w_geoid, h_county, jobs,
     sa01, sa02, sa03, se01, se02, se03, si01, si02, si03,
     h_lat, h_lon, w_lat, w_lon)
SELECT o.metro, o.h_geocode, o.w_geocode, left(o.h_geocode, 5), o.jobs,
       o.sa01, o.sa02, o.sa03, o.se01, o.se02, o.se03,
       o.si01, o.si02, o.si03,
       ST_Y(hb.pt), ST_X(hb.pt), ST_Y(wb.pt), ST_X(wb.pt)
FROM census.od_core o
JOIN census.blocks hb ON hb.geoid20 = o.h_geocode
JOIN census.blocks wb ON wb.geoid20 = o.w_geocode
WHERE o.metro = %s AND left(o.h_geocode, 5) = ANY(%s)
"""


def _selected_counties(cur, metro: str, coverage: float, margin: float = 0.05):
    """County FIPS kept by the coverage rule, plus a margin.

    Mirrors datasource.select_counties: the smallest prefix by workers desc
    whose cumulative share reaches `coverage`. The margin stores a slightly
    wider set than today's threshold selects, so nudging county_coverage does
    not force a re-derive -- the datasource filters to the graph's real county
    list at query time anyway.
    """
    cur.execute("""
        SELECT county_fips, workers FROM public.metro_county_flows
        WHERE metro = %s ORDER BY workers DESC
    """, (metro,))
    rows = cur.fetchall()
    total = sum(r[1] for r in rows)
    target = min(0.99, coverage + margin) * total
    keep, cum = [], 0
    for fips, workers in rows:
        keep.append(fips)
        cum += workers
        if cum >= target:
            break
    return keep


def phase_pairs(conn, metros) -> None:
    """Materialise the block pairs, restricted to counties in the road graph.

    Homes outside the graph's counties are deliberately NOT stored: the shell
    only covers the selected counties, so such a home would snap to whatever
    node happens to sit on the graph boundary. The long tail (a Manhattan job
    held by someone living in Oregon) is real LODES data but unusable geometry.
    """
    p = _params()
    coverage = p.METRO_PARAMS["county_coverage"]
    cur = conn.cursor()
    for metro in metros:
        t0 = time.time()
        keep = _selected_counties(cur, metro, coverage)
        cur.execute("DELETE FROM public.metro_od_pairs WHERE metro = %s", (metro,))
        cur.execute(PAIRS_SQL, (metro, keep))
        n = cur.rowcount

        # Within the kept counties, a missing home block means missing TIGER
        # data and is a real defect. Reported in JOBS, not rows: one dropped
        # row can carry many commuters.
        cur.execute("""
            SELECT coalesce(sum(o.jobs), 0)
            FROM census.od_core o
            LEFT JOIN census.blocks hb ON hb.geoid20 = o.h_geocode
            WHERE o.metro = %s AND left(o.h_geocode, 5) = ANY(%s)
              AND hb.geoid20 IS NULL
        """, (metro, keep))
        lost = cur.fetchone()[0]
        cur.execute("SELECT coalesce(sum(jobs),0) FROM public.metro_od_pairs "
                    "WHERE metro=%s", (metro,))
        kept_jobs = cur.fetchone()[0]
        cur.execute("SELECT coalesce(sum(jobs),0) FROM census.od_core WHERE metro=%s",
                    (metro,))
        all_jobs = cur.fetchone()[0]
        conn.commit()

        tail = all_jobs - kept_jobs - lost
        note = ""
        if lost:
            note += f"  MISSING BLOCKS: {lost:,} jobs"
        print(f"  {metro:8s} {len(keep):3d} counties  {n:9,d} pairs  "
              f"{kept_jobs:9,d} jobs kept  {tail:8,d} out-of-graph tail "
              f"({100.0 * kept_jobs / max(1, all_jobs):.1f}% coverage)  "
              f"{time.time() - t0:5.1f}s{note}")


# ── Phase: boundaries ────────────────────────────────────────────────────────

def phase_boundaries(conn, metros) -> None:
    """Write each flow county's TIGER polygon into the boundary cache.

    welfare_rs/metro.py builds the shell by calling geocode_polygon(geocode)
    for every selected county. Seeding the cache with the TIGER polygon (the
    same pattern db/fetch_core_polygons.py uses for the cores) means the graph
    is cut against the SAME county boundary whose FIPS produced the flows,
    instead of whatever Nominatim returns for a similar-looking name -- and it
    skips hundreds of rate-limited geocoder calls.
    """
    p = _params()
    cache_dir = p.GEO_PARAMS["cache_dir"]
    coverage = p.METRO_PARAMS["county_coverage"]
    bdir = os.path.join(cache_dir, "boundaries")
    os.makedirs(bdir, exist_ok=True)

    cur = conn.cursor()
    written = skipped = 0
    for metro in metros:
        fips_keep = _selected_counties(cur, metro, coverage)
        cur.execute("""
            SELECT county_fips, geocode FROM public.metro_county_flows
            WHERE metro = %s AND county_fips = ANY(%s)
        """, (metro, fips_keep))
        keep = cur.fetchall()

        for fips, geocode in keep:
            slug = hashlib.md5(geocode.strip().lower().encode("utf-8")).hexdigest()[:16]
            path = os.path.join(bdir, f"{slug}.json")
            if os.path.exists(path):
                skipped += 1
                continue
            # Simplified before caching. Raw TIGER counties are enormously
            # detailed (DC's 35 counties carry 194k vertices between them) and
            # metro.py unions them into ONE polygon that is handed to
            # ox.graph_from_polygon -- i.e. serialised into an Overpass query.
            # 0.001 deg (~110 m) cuts that ~37x while leaving county shape
            # intact. It only bounds which OSM nodes are downloaded and which
            # county pool a shell node lands in; agent homes are placed from
            # block coordinates, not from these polygons.
            cur.execute(
                "SELECT ST_AsGeoJSON(ST_Multi(ST_MakeValid("
                "  ST_SimplifyPreserveTopology(geom, 0.001)))) "
                "FROM census.counties WHERE geoid = %s", (fips,))
            got = cur.fetchone()
            if got is None:
                raise SystemExit(f"{metro}: no TIGER polygon for county {fips}")
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({"query": geocode, "geometry": json.loads(got[0])}, f)
            os.replace(tmp, path)
            written += 1
        print(f"  {metro:8s} {len(keep):4d} counties cached")
    print(f"\n  {written} polygons written, {skipped} already cached -> {bdir}")


PHASES = {
    "core-blocks": phase_core_blocks,
    "od-core": phase_od_core,
    "home-states": phase_home_states,
    "flows": phase_flows,
    "pairs": phase_pairs,
    "boundaries": phase_boundaries,
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("phase", choices=list(PHASES))
    ap.add_argument("--metros", nargs="*", default=None)
    args = ap.parse_args()

    print(f"phase: {args.phase}  (LODES {LODES_JT} {LODES_YEAR})", flush=True)
    with psycopg.connect(DSN) as conn:
        # Session-only. The pairs phase hash-joins millions of OD rows against
        # the block table twice (home end and work end); the default 4 MB
        # work_mem would spill both to disk.
        cur = conn.cursor()
        cur.execute("SET work_mem = '2GB'")
        cur.execute("SET maintenance_work_mem = '2GB'")
        metros = _metros(conn, args.metros)
        PHASES[args.phase](conn, metros)
    return 0


if __name__ == "__main__":
    sys.exit(main())

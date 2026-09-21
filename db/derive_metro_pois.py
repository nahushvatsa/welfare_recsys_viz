#!/usr/bin/env python3
"""Derive public.metro_pois: clip poi.pois_us to geo.metro_cores, map categories.

This is the Postgres equivalent of data/filter_metro_pois.py and reproduces its
``_extract()`` semantics exactly, so LocalDataSource and PostgresDataSource serve
the same rows:

* NAICS pass — the pipe-packed NAICS/name/placekey columns are split and walked
  in order; the FIRST code that maps wins, and the name/placekey are taken at
  the SAME index, falling back to index 1 when that index is out of range.
* Text pass — only for rows with no NAICS hit: the first keyword (in list order)
  that is a substring of ``lower(sub_category || ' ' || top_category)``, using
  the index-1 name/placekey.
* Rows with an empty placekey or name are dropped.
* Coordinates are rounded to 6 dp, matching the CSV writer's ``f"{v:.6f}"``.

The category maps are read from params.POI_PARAMS at run time rather than being
copied into SQL, so they cannot drift from the model.

Per-metro DELETE + INSERT in one transaction each, so a single metro can be
re-derived without touching the others.

    python3 db/derive_metro_pois.py [--metros nyc miami] [--dry-run]
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time

import psycopg

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DSN = os.environ.get("WELFARE_LOADER_DSN", "")  # "" -> libpq env vars

# `metro_pois` is what the app reads; `pois_us` is 10.1M rows of raw Placekey
# data. The join is index-assisted: ST_Contains carries an implicit && bbox
# check, which the GIST index on pois_us.geom serves.
DERIVE_SQL = """
WITH clipped AS (
    SELECT p.id,
           c.metro,
           string_to_array(p.naics_code,    '|') AS naics_a,
           string_to_array(p.location_name, '|') AS name_a,
           string_to_array(p.placekey,      '|') AS key_a,
           string_to_array(p.sub_category,  '|') AS sub_a,
           string_to_array(p.top_category,  '|') AS top_a,
           p.latitude, p.longitude
    FROM geo.metro_cores c
    JOIN poi.pois_us p ON ST_Contains(c.geom, p.geom)
    WHERE c.metro = %(metro)s
      AND p.latitude  IS NOT NULL
      AND p.longitude IS NOT NULL
      AND p.placekey  IS NOT NULL
      AND p.location_name IS NOT NULL
),
sized AS (
    SELECT cl.*,
           -- The roster of businesses we can address by position at all.
           LEAST(coalesce(array_length(cl.name_a, 1), 0),
                 coalesce(array_length(cl.key_a,  1), 0)) AS n_pair,
           coalesce(array_length(cl.naics_a, 1), 0) AS n_naics,
           coalesce(array_length(cl.sub_a,   1), 0) AS n_sub,
           coalesce(array_length(cl.top_a,   1), 0) AS n_top
    FROM clipped cl
),
-- Shared-building rows pipe-pack several businesses into one record. Emit ONE
-- candidate per position rather than one per row, so a building with three
-- restaurants yields three POIs instead of one.
--
-- The industry-code list is deduplicated upstream (verified: 0 repeated codes,
-- and never longer than the name list), so on shared rows it is SHORTER than
-- the roster. Positions past its end have no category signal and fall out
-- below -- which keeps LEAST(n_names, n_codes) POIs, the most that can be
-- attributed. Unpacked rows have every length = 1 and are unaffected.
paired AS (
    SELECT s.id, s.metro, i.ord,
           btrim(s.name_a[i.ord]) AS name,
           btrim(s.key_a[i.ord])  AS place_id,
           CASE WHEN i.ord <= s.n_naics
                THEN split_part(btrim(s.naics_a[i.ord]), '.', 1) END AS code,
           lower(coalesce(CASE WHEN i.ord <= s.n_sub THEN s.sub_a[i.ord] END, '')
                 || ' '
                 || coalesce(CASE WHEN i.ord <= s.n_top THEN s.top_a[i.ord] END, ''))
               AS cat_text,
           s.latitude, s.longitude
    FROM sized s
    CROSS JOIN LATERAL generate_series(1, s.n_pair) AS i(ord)
),
-- NAICS wins; the text rules are consulted only for positions it did not map,
-- and against that position's OWN sub/top category, never the whole record.
classified AS (
    SELECT p.id, p.ord, p.metro, p.place_id, p.name, p.latitude, p.longitude,
           coalesce(nm.category, tm.category) AS category
    FROM paired p
    LEFT JOIN naics_map nm ON nm.code = p.code
    LEFT JOIN LATERAL (
        SELECT t.category
        FROM text_map t
        WHERE nm.category IS NULL
          AND strpos(p.cat_text, t.keyword) > 0
        ORDER BY t.ord
        LIMIT 1
    ) tm ON true
    WHERE p.place_id <> '' AND p.name <> ''
)
INSERT INTO public.metro_pois (metro, place_id, name, category, latitude, longitude)
-- A placekey can recur across parent/child rows; the PK forbids duplicates, so
-- keep the lowest (id, position) deterministically.
SELECT DISTINCT ON (c.metro, c.place_id)
       c.metro, c.place_id, c.name, c.category,
       round(c.latitude::numeric,  6)::double precision,
       round(c.longitude::numeric, 6)::double precision
FROM classified c
WHERE c.category IS NOT NULL
ORDER BY c.metro, c.place_id, c.id, c.ord
"""


def _params():
    """Load params.py standalone (welfare_rs/__init__ imports the geo stack)."""
    spec = importlib.util.spec_from_file_location(
        "wrs_params", os.path.join(REPO, "welfare_rs", "params.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_maps(cur, p) -> None:
    """Materialise POI_PARAMS' category maps as temp tables for the join."""
    cur.execute("CREATE TEMP TABLE naics_map (code text PRIMARY KEY, "
                "category text NOT NULL) ON COMMIT DROP")
    cur.execute("CREATE TEMP TABLE text_map (ord int PRIMARY KEY, keyword text "
                "NOT NULL, category text NOT NULL) ON COMMIT DROP")

    naics = list(p.POI_PARAMS["naics_to_category"].items())
    with cur.copy("COPY naics_map (code, category) FROM STDIN") as copy:
        for code, category in naics:
            copy.write_row((str(code), category))

    # Order is significant: _extract() returns the FIRST keyword that matches.
    text = list(p.POI_PARAMS["text_to_category"])
    with cur.copy("COPY text_map (ord, keyword, category) FROM STDIN") as copy:
        for i, (keyword, category) in enumerate(text):
            copy.write_row((i, keyword.lower(), category))

    cur.execute("ANALYZE naics_map")
    cur.execute("ANALYZE text_map")
    return len(naics), len(text)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--metros", nargs="*", default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="Derive, report counts, then roll back.")
    args = ap.parse_args()

    p = _params()
    metros = args.metros or list(p.METRO_PARAMS["metros"])
    unknown = [m for m in metros if m not in p.METRO_PARAMS["metros"]]
    if unknown:
        raise SystemExit(f"Unknown metro key(s): {unknown}")

    valid_categories = set(p.CATEGORY_KEYWORDS)
    total = 0
    with psycopg.connect(DSN) as conn:
        cur = conn.cursor()
        n_naics, n_text = _load_maps(cur, p)
        print(f"category maps: {n_naics} NAICS codes, {n_text} text rules")

        for metro in metros:
            cur.execute("SELECT 1 FROM geo.metro_cores WHERE metro = %s", (metro,))
            if cur.fetchone() is None:
                raise SystemExit(
                    f"No core polygon for {metro!r} -- run db/fetch_core_polygons.py first")

            t0 = time.time()
            cur.execute("DELETE FROM public.metro_pois WHERE metro = %s", (metro,))
            deleted = cur.rowcount
            cur.execute(DERIVE_SQL, {"metro": metro})
            kept = cur.rowcount
            total += kept

            cur.execute(
                "SELECT category, count(*) FROM public.metro_pois "
                "WHERE metro = %s GROUP BY category ORDER BY count(*) DESC", (metro,))
            breakdown = cur.fetchall()
            bad = [c for c, _ in breakdown if c not in valid_categories]
            if bad:
                raise SystemExit(f"{metro}: categories outside CATEGORY_KEYWORDS: {bad}")

            note = f" (replaced {deleted:,})" if deleted else ""
            print(f"  {metro:8s} {kept:7,d} POIs{note}  {time.time() - t0:5.1f}s  "
                  + " ".join(f"{c}:{n:,}" for c, n in breakdown))

        if args.dry_run:
            conn.rollback()
            print(f"\nDRY RUN -- rolled back ({total:,} rows would have been written)")
        else:
            conn.commit()
            print(f"\nOK: {total:,} rows in public.metro_pois")
    return 0


if __name__ == "__main__":
    sys.exit(main())

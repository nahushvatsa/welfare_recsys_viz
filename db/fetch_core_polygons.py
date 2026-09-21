#!/usr/bin/env python3
"""Fetch each metro's core-city boundary from Nominatim into geo.metro_core_parts.

Writes each polygon to TWO places, on purpose:

1. ``geo.metro_core_parts`` in Postgres — what the ETL clips POIs against and
   what the LODES flow query will test work blocks against.
2. ``<cache>/boundaries/<md5>.json`` — the exact file format
   ``welfare_rs.metro.geocode_polygon`` reads, so it finds these on its first
   call and never re-geocodes.

Both must hold the SAME polygon: if the POI clip and the network's core layer
disagree, POIs land outside the only layer permitted to hold them.

Deliberately does not import welfare_rs (that pulls osmnx/geopandas, which
aren't installed yet). It replicates ``geocode_to_gdf``'s selection — the first
result carrying a (Multi)Polygon — against the same query strings from
params.METRO_PARAMS.

Idempotent: existing cache files and existing rows are left alone unless
--refetch is passed. Honours Nominatim's 1 req/s usage policy.

    python3 db/fetch_core_polygons.py [--metros nyc miami] [--refetch]
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
import urllib.parse
import urllib.request

import psycopg

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOMINATIM = "https://nominatim.openstreetmap.org/search"
# Nominatim blocks requests without a genuine identifying User-Agent.
UA = "welfare-rs-etl/0.1 (+https://airecsim.cusp.nyu.edu)"
SLEEP_S = 1.1

# Sanity envelope in square degrees, so a wrong Nominatim hit cannot silently
# become the clip polygon. Manhattan ~0.006, Los Angeles ~0.13.
AREA_MIN, AREA_MAX = 0.0005, 1.0

DSN = os.environ.get("WELFARE_LOADER_DSN", "")  # "" -> libpq env vars


def _params():
    """Load params.py standalone (welfare_rs/__init__ imports the geo stack)."""
    spec = importlib.util.spec_from_file_location(
        "wrs_params", os.path.join(REPO, "welfare_rs", "params.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _cache_path(cache_dir: str, query: str) -> str:
    # Must match geocode_polygon(): md5 of the stripped, lowercased query, 16 hex.
    slug = hashlib.md5(query.strip().lower().encode("utf-8")).hexdigest()[:16]
    d = os.path.join(cache_dir, "boundaries")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{slug}.json")


def _ring_area(ring) -> float:
    """|shoelace| in square degrees — a scale check, not a true area."""
    total = 0.0
    for i in range(len(ring) - 1):
        total += ring[i][0] * ring[i + 1][1] - ring[i + 1][0] * ring[i][1]
    return abs(total) / 2.0


def _geom_area(geom) -> float:
    if geom["type"] == "Polygon":
        return _ring_area(geom["coordinates"][0])
    return sum(_ring_area(poly[0]) for poly in geom["coordinates"])


def _bounds(geom):
    polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
    xs = [pt[0] for poly in polys for ring in poly for pt in ring]
    ys = [pt[1] for poly in polys for ring in poly for pt in ring]
    return min(xs), min(ys), max(xs), max(ys)


def fetch(query: str) -> dict:
    url = NOMINATIM + "?" + urllib.parse.urlencode({
        "q": query, "format": "json", "limit": "5", "polygon_geojson": "1",
    })
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as resp:
        results = json.load(resp)
    for r in results:
        geom = r.get("geojson") or {}
        if geom.get("type") in ("Polygon", "MultiPolygon"):
            return {"geometry": geom, "display_name": r.get("display_name", "")}
    raise SystemExit(f"No (Multi)Polygon result for {query!r}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--metros", nargs="*", default=None)
    ap.add_argument("--refetch", action="store_true",
                    help="Ignore cached boundary files and re-query Nominatim.")
    args = ap.parse_args()

    p = _params()
    cache_dir = p.GEO_PARAMS["cache_dir"]
    metros = args.metros or list(p.METRO_PARAMS["metros"])

    calls = 0
    with psycopg.connect(DSN) as conn:
        cur = conn.cursor()
        for metro in metros:
            spec = p.METRO_PARAMS["metros"][metro]
            for query in spec["core_places"]:
                path = _cache_path(cache_dir, query)
                display = None

                if os.path.exists(path) and not args.refetch:
                    with open(path, encoding="utf-8") as f:
                        geom = json.load(f)["geometry"]
                    origin = "cache"
                else:
                    if calls:
                        time.sleep(SLEEP_S)
                    got = fetch(query)
                    calls += 1
                    geom, display = got["geometry"], got["display_name"]
                    origin = "nominatim"

                area = _geom_area(geom)
                if not (AREA_MIN <= area <= AREA_MAX):
                    raise SystemExit(
                        f"{query!r} -> implausible extent {area:.5f} deg^2"
                        f"{' (' + display + ')' if display else ''}; refusing to store")

                if origin == "nominatim":
                    # Temp file + rename: an interrupted run must not leave a
                    # partial entry that geocode_polygon would later trust.
                    tmp = path + ".tmp"
                    with open(tmp, "w", encoding="utf-8") as f:
                        json.dump({"query": query, "geometry": geom}, f)
                    os.replace(tmp, path)

                # ST_MakeValid: OSM admin boundaries occasionally self-intersect,
                # which would make ST_Contains unreliable. ST_Multi normalises
                # single Polygons to match the column's declared type.
                cur.execute(
                    """
                    INSERT INTO geo.metro_core_parts
                        (metro, place_query, display_name, geom)
                    VALUES (%s, %s, %s,
                            ST_Multi(ST_MakeValid(
                                ST_SetSRID(ST_GeomFromGeoJSON(%s), 4326))))
                    ON CONFLICT (metro, place_query) DO UPDATE
                        SET display_name = COALESCE(EXCLUDED.display_name,
                                                    geo.metro_core_parts.display_name),
                            geom = EXCLUDED.geom,
                            fetched_at = now()
                    """,
                    (metro, query, display, json.dumps(geom)),
                )
                w, s, e, n = _bounds(geom)
                print(f"  {metro:8s} {origin:9s} {query}")
                if display:
                    print(f"           -> {display[:74]}")
                print(f"           -> {geom['type']}, {area:.5f} deg^2, "
                      f"bbox ({w:.4f},{s:.4f},{e:.4f},{n:.4f})")

        # Rebuild the unioned per-metro polygon from the parts just stored.
        for metro in metros:
            cur.execute(
                """
                INSERT INTO geo.metro_cores (metro, label, parts, geom)
                SELECT %s, %s, count(*)::int,
                       ST_Multi(ST_MakeValid(ST_Union(geom)))
                FROM geo.metro_core_parts WHERE metro = %s
                ON CONFLICT (metro) DO UPDATE
                    SET label = EXCLUDED.label,
                        parts = EXCLUDED.parts,
                        geom  = EXCLUDED.geom
                """,
                (metro, p.METRO_PARAMS["metros"][metro]["label"], metro),
            )
        conn.commit()

    print(f"\nOK: {calls} Nominatim call(s); geo.metro_core_parts + geo.metro_cores updated.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Download TIGER geography and load it into the `census` schema.

Three layers, loaded independently:

    python3 db/fetch_tiger.py states              # national, ~1 min
    python3 db/fetch_tiger.py counties            # national, ~1 min
    python3 db/fetch_tiger.py blocks --states 11 24 51

VINTAGE MATTERS. LODES8 keys on **2020** Census blocks, so the block layer must
be TABBLOCK20 (the 2020 blocks). A block's 15-digit GEOID20 embeds its county
(chars 1-5), and we take county assignment from that prefix rather than from a
spatial join -- the prefix is authoritative and cannot drift from LODES.

Idempotent per state: an already-loaded state is skipped unless --reload is
passed, and downloads are reused unless --redownload is passed.

Connection comes from libpq env vars (PGHOST/PGDATABASE/PGUSER/PGPASSWORD) or a
full DSN in $WELFARE_LOADER_DSN.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import psycopg

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DSN = os.environ.get("WELFARE_LOADER_DSN", "")  # "" -> libpq env vars
DL_DIR = os.path.join(REPO, "data", "census", "tiger")

TIGER_YEAR = os.environ.get("TIGER_YEAR", "2023")
BASE = f"https://www2.census.gov/geo/tiger/TIGER{TIGER_YEAR}"

# Read blocks in chunks so a 900k-block state never materialises whole.
CHUNK = 50_000


def _download(url: str, path: str, redownload: bool = False) -> str:
    """Stream a URL to disk (skip when already present and non-empty)."""
    import requests

    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path) and os.path.getsize(path) > 0 and not redownload:
        print(f"    cached  {os.path.basename(path)} "
              f"({os.path.getsize(path) / 1e6:.1f} MB)", flush=True)
        return path
    tmp = path + ".part"
    t0 = time.time()
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done = 0
        with open(tmp, "wb") as f:
            for block in r.iter_content(chunk_size=1 << 20):
                f.write(block)
                done += len(block)
    os.replace(tmp, path)  # atomic: a partial file must never look complete
    print(f"    fetched {os.path.basename(path)} ({done / 1e6:.1f} MB, "
          f"{time.time() - t0:.0f}s)", flush=True)
    return path


def _read_chunks(zip_path: str, columns, chunk: int = CHUNK):
    """Yield (GeoDataFrame, offset) chunks from a zipped shapefile."""
    import pyogrio

    src = f"/vsizip/{os.path.abspath(zip_path)}"
    total = pyogrio.read_info(src)["features"]
    offset = 0
    while offset < total:
        gdf = pyogrio.read_dataframe(
            src, columns=list(columns), skip_features=offset, max_features=chunk,
        )
        if len(gdf) == 0:
            break
        yield gdf, offset, total
        offset += len(gdf)


def _copy_geom(cur, target: str, cols: list, rows_iter,
               geom_cols=("geom",), srid: int = 4326,
               on_conflict: str = "") -> int:
    """COPY rows carrying hex-WKB geometry into `target`.

    COPY cannot call functions, and plain WKB carries no SRID, so a typed
    geometry(...,4326) column would reject it outright. We stage into a TEMP
    table with every column as text and let one INSERT..SELECT apply
    ST_SetSRID to the geometry columns.
    """
    stage = "stage_" + target.replace(".", "_")
    cur.execute(f"DROP TABLE IF EXISTS {stage}")
    coldefs = ", ".join(f"{c} text" for c in cols)
    cur.execute(f"CREATE TEMP TABLE {stage} ({coldefs})")

    n = 0
    with cur.copy(f"COPY {stage} ({', '.join(cols)}) FROM STDIN") as copy:
        for row in rows_iter:
            copy.write_row(row)
            n += 1

    # Everything staged as text, so each non-geometry column needs an explicit
    # cast back to whatever the destination declares (bigint, integer, char(n)).
    cur.execute(
        "SELECT attname, format_type(atttypid, atttypmod) FROM pg_attribute "
        "WHERE attrelid = %s::regclass AND attnum > 0 AND NOT attisdropped",
        (target,),
    )
    coltype = dict(cur.fetchall())
    select_cols = [
        (f"ST_SetSRID(ST_GeomFromWKB(decode({c}, 'hex')), {srid})"
         if c in geom_cols else f"{c}::{coltype[c]}")
        for c in cols
    ]
    cur.execute(
        f"INSERT INTO {target} ({', '.join(cols)}) "
        f"SELECT {', '.join(select_cols)} FROM {stage} {on_conflict}"
    )
    inserted = cur.rowcount
    cur.execute(f"DROP TABLE {stage}")
    return inserted if on_conflict else n


def _hexwkb(geom) -> str | None:
    """Hex WKB, forced to MultiPolygon so it matches the column type."""
    import shapely

    if geom is None or geom.is_empty:
        return None
    if geom.geom_type == "Polygon":
        geom = shapely.MultiPolygon([geom])
    return shapely.to_wkb(geom, hex=True)


# ── Layers ───────────────────────────────────────────────────────────────────

def load_states(conn, redownload: bool) -> None:
    url = f"{BASE}/STATE/tl_{TIGER_YEAR}_us_state.zip"
    path = _download(url, os.path.join(DL_DIR, os.path.basename(url)), redownload)
    cols = ["STATEFP", "STUSPS", "NAME", "geometry"]

    cur = conn.cursor()
    cur.execute("TRUNCATE census.states")
    total = 0
    for gdf, _off, _tot in _read_chunks(path, cols):
        rows = (
            (r.STATEFP, r.STUSPS, r.NAME, _hexwkb(r.geometry))
            for r in gdf.itertuples()
        )
        total += _copy_geom(cur, "census.states",
                            ["statefp", "usps", "name", "geom"], rows)
    conn.commit()
    print(f"  census.states: {total:,} rows")


def load_counties(conn, redownload: bool, supplement: bool = False) -> None:
    """Load the national county file.

    ``supplement`` inserts ONLY county GEOIDs not already present, which is how
    a second vintage fills a gap without disturbing the first. Connecticut is
    the reason this exists: it replaced counties with planning regions in 2022,
    so TIGER2023 lists 09110-09190 while 2020 Census blocks -- and therefore
    LODES -- still carry the old 09001-09015. Supplementing from TIGER2021
    restores names and polygons for those old counties, matching the block
    GEOIDs actually present in the data.
    """
    url = f"{BASE}/COUNTY/tl_{TIGER_YEAR}_us_county.zip"
    path = _download(url, os.path.join(DL_DIR, os.path.basename(url)), redownload)
    cols = ["GEOID", "STATEFP", "NAME", "NAMELSAD", "ALAND", "AWATER", "geometry"]

    cur = conn.cursor()
    if not supplement:
        cur.execute("TRUNCATE census.counties")
    total = 0
    for gdf, _off, _tot in _read_chunks(path, cols):
        rows = (
            (r.GEOID, r.STATEFP, r.NAME, r.NAMELSAD,
             str(int(r.ALAND)), str(int(r.AWATER)), _hexwkb(r.geometry))
            for r in gdf.itertuples()
        )
        total += _copy_geom(
            cur, "census.counties",
            ["geoid", "statefp", "name", "namelsad", "aland", "awater", "geom"], rows,
            on_conflict="ON CONFLICT (geoid) DO NOTHING" if supplement else "")
    conn.commit()
    verb = "added" if supplement else "rows"
    print(f"  census.counties: {total:,} {verb} (TIGER{TIGER_YEAR})")


def _loaded_states(conn) -> set:
    cur = conn.cursor()
    cur.execute("SELECT DISTINCT statefp FROM census.blocks")
    return {r[0] for r in cur.fetchall()}


def load_blocks(conn, states: list, redownload: bool, reload_: bool) -> None:
    have = _loaded_states(conn)
    cols = ["GEOID20", "STATEFP20", "COUNTYFP20", "ALAND20", "AWATER20",
            "POP20", "HOUSING20", "INTPTLAT20", "INTPTLON20", "geometry"]
    dest = ["geoid20", "statefp", "county_fips", "aland", "awater",
            "pop20", "housing20", "pt", "geom"]

    for st in states:
        if st in have and not reload_:
            cur = conn.cursor()
            cur.execute("SELECT count(*) FROM census.blocks WHERE statefp = %s", (st,))
            print(f"  {st}: already loaded ({cur.fetchone()[0]:,} blocks) — skipping")
            continue

        url = f"{BASE}/TABBLOCK20/tl_{TIGER_YEAR}_{st}_tabblock20.zip"
        print(f"  {st}:", flush=True)
        path = _download(url, os.path.join(DL_DIR, os.path.basename(url)), redownload)

        cur = conn.cursor()
        cur.execute("DELETE FROM census.blocks WHERE statefp = %s", (st,))
        t0, total = time.time(), 0
        for gdf, off, tot in _read_chunks(path, cols):
            def _rows(gdf=gdf):
                import shapely

                for r in gdf.itertuples():
                    lat, lon = float(r.INTPTLAT20), float(r.INTPTLON20)
                    yield (
                        r.GEOID20,
                        r.STATEFP20,
                        r.STATEFP20 + r.COUNTYFP20,
                        str(int(r.ALAND20)),
                        str(int(r.AWATER20)),
                        str(int(r.POP20)),
                        str(int(r.HOUSING20)),
                        shapely.to_wkb(shapely.Point(lon, lat), hex=True),
                        _hexwkb(r.geometry),
                    )

            # pt is a Point column; both geometry columns need SRID applied.
            n = _copy_geom(cur, "census.blocks", dest, _rows(),
                           geom_cols=("pt", "geom"))
            total += n
            print(f"    {off + n:,}/{tot:,} blocks", end="\r", flush=True)
        conn.commit()
        print(f"    {total:,} blocks in {time.time() - t0:.0f}s          ")



def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("layer", choices=["states", "counties", "blocks"])
    ap.add_argument("--states", nargs="*", default=None,
                    help="2-digit state FIPS (blocks layer only).")
    ap.add_argument("--redownload", action="store_true")
    ap.add_argument("--reload", action="store_true",
                    help="Re-load states already present in census.blocks.")
    ap.add_argument("--supplement", action="store_true",
                    help="counties: insert only GEOIDs not already present "
                         "(fill an older vintage's counties, e.g. pre-2022 CT).")
    args = ap.parse_args()

    print(f"TIGER{TIGER_YEAR} -> census schema", flush=True)
    with psycopg.connect(DSN) as conn:
        if args.layer == "states":
            load_states(conn, args.redownload)
        elif args.layer == "counties":
            load_counties(conn, args.redownload, args.supplement)
        else:
            if not args.states:
                raise SystemExit("blocks layer needs --states <fips> [...]")
            states = [s.zfill(2) for s in args.states]
            load_blocks(conn, states, args.redownload, args.reload)
    return 0


if __name__ == "__main__":
    sys.exit(main())

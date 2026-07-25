"""Filter the large all-US POI file to per-metro principal-city leisure POIs.

One streaming pass over the raw Placekey/NAICS table (``poi_children_merged_
by_wkt.csv``, ~375 MB) clips rows to each metro's **core-city polygon** (the
POI bound of the two-layer model — Manhattan for nyc, SF+Oakland for sf_bay,
…) and maps NAICS codes / category text to the recommender's leisure
categories. Output: one small CSV per metro
(``place_id, name, category, latitude, longitude``) under
``<cache>/pois/<metro>_leisure_pois.csv`` — exactly what
``LocalDataSource.poi_rows`` serves and what the ``metro_pois`` Postgres table
will hold on the server. All outputs are local data (gitignored).

Some rows are "shared polygons" packing several businesses with
``|``-separated NAICS/names/placekeys; we align by index and keep the first
leisure match.

Usage
-----
    python data/filter_metro_pois.py                       # all 8 metros
    python data/filter_metro_pois.py --metros miami seattle
    python data/filter_metro_pois.py --source /path/to/raw.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shapely  # noqa: E402

from welfare_rs import params  # noqa: E402
from welfare_rs.metro import geocode_polygon  # noqa: E402

_USECOLS = [
    "LATITUDE", "LONGITUDE", "CITY", "LOCATION_NAME",
    "PLACEKEY", "NAICS_CODE", "SUB_CATEGORY", "TOP_CATEGORY",
]


def _extract(row, naics_map, text_map):
    """Return (place_id, name, category) for the first leisure match, or None."""
    naics_list = str(row["NAICS_CODE"]).split("|")
    names = str(row["LOCATION_NAME"]).split("|")
    keys = str(row["PLACEKEY"]).split("|")
    for i, code in enumerate(naics_list):
        code = code.strip().split(".")[0]
        category = naics_map.get(code)
        if category:
            name = names[i].strip() if i < len(names) else names[0].strip()
            key = keys[i].strip() if i < len(keys) else keys[0].strip()
            return key, name, category
    text = (str(row["SUB_CATEGORY"]) + " " + str(row["TOP_CATEGORY"])).lower()
    for keyword, category in text_map:
        if keyword in text:
            return keys[0].strip(), names[0].strip(), category
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "poi_children_merged_by_wkt.csv"))
    ap.add_argument("--metros", nargs="*", default=None,
                    help="Metro keys (default: all in params.METRO_PARAMS).")
    ap.add_argument("--out-dir", default=os.path.join(params.GEO_PARAMS["cache_dir"], "pois"))
    ap.add_argument("--chunksize", type=int, default=200_000)
    args = ap.parse_args()

    metros = args.metros or list(params.METRO_PARAMS["metros"])
    naics_map = params.POI_PARAMS["naics_to_category"]
    text_map = params.POI_PARAMS["text_to_category"]
    os.makedirs(args.out_dir, exist_ok=True)

    # Core polygons (geocoded once, disk-cached) + bbox prefilters.
    cores, boxes = {}, {}
    for metro in metros:
        places = params.METRO_PARAMS["metros"][metro]["core_places"]
        geom = shapely.union_all([geocode_polygon(q) for q in places])
        shapely.prepare(geom)
        cores[metro] = geom
        boxes[metro] = geom.bounds  # (west, south, east, north)
        print(f"{metro}: core polygon ready ({', '.join(places)})")

    writers, files = {}, {}
    for metro in metros:
        path = os.path.join(args.out_dir, f"{metro}_leisure_pois.csv")
        files[metro] = open(path, "w", newline="", encoding="utf-8")
        writers[metro] = csv.writer(files[metro])
        writers[metro].writerow(["place_id", "name", "category", "latitude", "longitude"])

    scanned = 0
    kept: Counter = Counter()
    try:
        reader = pd.read_csv(
            args.source, usecols=_USECOLS, chunksize=args.chunksize,
            dtype=str, low_memory=False,
        )
        for chunk in reader:
            scanned += len(chunk)
            lat = pd.to_numeric(chunk["LATITUDE"], errors="coerce").to_numpy()
            lon = pd.to_numeric(chunk["LONGITUDE"], errors="coerce").to_numpy()
            valid = ~(np.isnan(lat) | np.isnan(lon))
            for metro in metros:
                west, south, east, north = boxes[metro]
                mask = valid & (lat >= south) & (lat <= north) & (lon >= west) & (lon <= east)
                if not mask.any():
                    continue
                mask[mask] = shapely.contains_xy(cores[metro], lon[mask], lat[mask])
                if not mask.any():
                    continue
                sub = chunk[mask]
                for (_, row), la, lo in zip(sub.iterrows(), lat[mask], lon[mask]):
                    got = _extract(row, naics_map, text_map)
                    if got is None:
                        continue
                    place_id, name, category = got
                    if not place_id or not name:
                        continue
                    writers[metro].writerow([place_id, name, category, f"{la:.6f}", f"{lo:.6f}"])
                    kept[metro] += 1
            print(f"  scanned {scanned:,} rows · kept " +
                  " ".join(f"{m}:{kept[m]:,}" for m in metros), flush=True)
    finally:
        for f in files.values():
            f.close()
    # Header-only outputs would read as "real POIs available" downstream;
    # drop them so those metros cleanly fall back to synthetic POIs.
    for metro in metros:
        if kept[metro] == 0:
            os.remove(os.path.join(args.out_dir, f"{metro}_leisure_pois.csv"))

    print(f"\nDONE -> {args.out_dir}")
    for metro in metros:
        note = "" if kept[metro] else "  (no coverage in source -> synthetic fallback)"
        print(f"  {metro}: {kept[metro]:,} leisure POIs{note}")


if __name__ == "__main__":
    main()

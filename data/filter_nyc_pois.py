"""Filter the large all-US POI file down to New York leisure POIs.

The raw dataset (``poi_children_merged_by_wkt.csv``, ~375 MB) is a Placekey/NAICS
POI table. This script streams it in chunks (skipping the huge ``POLYGON_WKT``
column), keeps only rows inside the NYC bounding box whose NAICS code or category
text maps to one of the recommender's leisure categories, and writes a small CSV
(``place_id, name, category, latitude, longitude``).

Some rows are "shared polygons" packing several businesses with ``|``-separated
NAICS/names/placekeys; we align by index and keep the first leisure match.

Usage
-----
    python data/filter_nyc_pois.py \
        --source ~/Desktop/poi_children_merged_by_wkt.csv \
        --out    <params.NYC_POI_CSV_PATH>            # default
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402

from welfare_rs import params  # noqa: E402

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
    ap.add_argument("--source", default=os.path.expanduser("~/Desktop/poi_children_merged_by_wkt.csv"))
    ap.add_argument("--out", default=params.NYC_POI_CSV_PATH)
    ap.add_argument("--chunksize", type=int, default=200_000)
    args = ap.parse_args()

    west, south, east, north = params.POI_PARAMS["nyc_filter_bbox"]
    naics_map = params.POI_PARAMS["naics_to_category"]
    text_map = params.POI_PARAMS["text_to_category"]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    scanned = kept = 0
    from collections import Counter
    cat_counts: Counter = Counter()

    with open(args.out, "w", newline="", encoding="utf-8") as fout:
        writer = csv.writer(fout)
        writer.writerow(["place_id", "name", "category", "latitude", "longitude"])

        reader = pd.read_csv(
            args.source, usecols=_USECOLS, chunksize=args.chunksize,
            dtype=str, low_memory=False,
        )
        for chunk in reader:
            scanned += len(chunk)
            lat = pd.to_numeric(chunk["LATITUDE"], errors="coerce")
            lon = pd.to_numeric(chunk["LONGITUDE"], errors="coerce")
            mask = (lat >= south) & (lat <= north) & (lon >= west) & (lon <= east)
            sub = chunk[mask]
            for (_, row), la, lo in zip(sub.iterrows(), lat[mask], lon[mask]):
                got = _extract(row, naics_map, text_map)
                if got is None:
                    continue
                place_id, name, category = got
                if not place_id or not name:
                    continue
                writer.writerow([place_id, name, category, f"{la:.6f}", f"{lo:.6f}"])
                kept += 1
                cat_counts[category] += 1
            print(f"  scanned {scanned:,} rows · kept {kept:,} NYC leisure POIs", flush=True)

    print(f"\nDONE: {kept:,} POIs -> {args.out}")
    print("by category:", dict(cat_counts.most_common()))


if __name__ == "__main__":
    main()

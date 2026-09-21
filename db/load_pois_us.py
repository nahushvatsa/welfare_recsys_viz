#!/usr/bin/env python3
"""Stream data/pois.parquet into poi.pois_us via COPY (pyarrow -> psycopg3).

Reads only the columns we keep (skips the heavy POLYGON_WKT). Connection comes
from standard libpq env vars (PGHOST/PGPORT/PGDATABASE/PGUSER/PGPASSWORD) or a
full DSN in $WELFARE_LOADER_DSN. Run from the repo root, ideally inside tmux:

    python3 db/load_pois_us.py
"""
from __future__ import annotations

import os
import time

import psycopg
import pyarrow.parquet as pq

PARQUET = os.environ.get("POIS_PARQUET", "data/pois.parquet")
DSN = os.environ.get("WELFARE_LOADER_DSN", "")  # "" -> libpq env vars
BATCH = 65536

# parquet column -> table column (order defines the COPY column list).
COLS = [
    ("PLACEKEY", "placekey"),
    ("PARENT_PLACEKEY", "parent_placekey"),
    ("LOCATION_NAME", "location_name"),
    ("NAICS_CODE", "naics_code"),
    ("TOP_CATEGORY", "top_category"),
    ("SUB_CATEGORY", "sub_category"),
    ("TOP_CATEGORY_2DIGIT", "top_category_2digit"),
    ("TOP_CATEGORY_FINAL", "top_category_final"),
    ("STATE", "state"),
    ("CITY", "city"),
    ("POSTAL_CODE", "postal_code"),
    ("STREET_ADDRESS", "street_address"),
    ("LATITUDE", "latitude"),
    ("LONGITUDE", "longitude"),
    ("POLYGON_CLASS", "polygon_class"),
    ("ENCLOSED", "enclosed"),
    ("INCLUDES_PARKING_LOT", "includes_parking_lot"),
    ("GEOMETRY_TYPE", "geometry_type"),
    ("WKT_AREA_SQ_METERS", "wkt_area_sq_meters"),
]
SRC = [c[0] for c in COLS]
DST = [c[1] for c in COLS]


def main() -> None:
    pf = pq.ParquetFile(PARQUET)
    total = pf.metadata.num_rows
    copy_sql = "COPY poi.pois_us (" + ", ".join(DST) + ") FROM STDIN"
    print(f"Loading {total:,} rows from {PARQUET} -> poi.pois_us", flush=True)

    done, t0 = 0, time.time()
    with psycopg.connect(DSN) as conn:
        with conn.cursor() as cur, cur.copy(copy_sql) as copy:
            for batch in pf.iter_batches(batch_size=BATCH, columns=SRC):
                data = {name: batch.column(name).to_pylist() for name in SRC}
                for i in range(batch.num_rows):
                    copy.write_row(tuple(data[s][i] for s in SRC))
                done += batch.num_rows
                rate = done / max(time.time() - t0, 1e-6)
                print(f"  {done:,}/{total:,} ({done * 100 // total}%)  {rate:,.0f} rows/s",
                      flush=True)
        # connection context commits on clean exit
    print(f"Done: {done:,} rows in {time.time() - t0:,.0f}s", flush=True)


if __name__ == "__main__":
    main()

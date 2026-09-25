"""Generate a New York persona file from the existing Seattle personas.

Every behavioural attribute is copied verbatim from the source file; only each
persona's home location (``start_latitude`` / ``start_longitude``) is replaced
with a real node sampled from the New York road network, so the behavioural
population is identical to the paper's while the geography becomes New York.

Usage
-----
    python data/make_nyc_personas.py                       # -> params.NYC_PERSONA_CSV_PATH
    python data/make_nyc_personas.py --city nyc_midtown --out /tmp/nyc.csv --seed 7

Note: the first run downloads and caches the chosen city's OSM network.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys

# Allow running from the data/ directory by putting the repo root on the path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from welfare_rs import params  # noqa: E402
from welfare_rs.geo import build_road_network  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default=params.DEFAULT_PERSONA_CSV_PATH,
                    help="Source persona CSV (Seattle).")
    ap.add_argument("--city", default=params.GEO_PARAMS["default_city"],
                    help="City preset from params.GEO_PARAMS['cities'].")
    ap.add_argument("--out", default=params.NYC_PERSONA_CSV_PATH,
                    help="Output persona CSV path.")
    ap.add_argument("--seed", type=int, default=2025)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    net = build_road_network(args.city, active=False)  # placement only, never simulated
    print(f"network: {net}")

    with open(args.source, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = [r for r in reader if r.get("PersonaID")]

    for row in rows:
        lat, lon = net.sample_node_latlon(rng)
        row["start_latitude"] = f"{lat:.6f}"
        row["start_longitude"] = f"{lon:.6f}"

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} New York personas -> {args.out}")


if __name__ == "__main__":
    main()

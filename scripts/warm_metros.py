"""Prefetch and warm-cache the two-layer road networks for all metros.

For each metro this geocodes the boundary polygons (Nominatim), downloads the
core drive network + arterial shell (Overpass), composes/simplifies them, and
writes all three disk caches (boundaries / GraphML / warm pickle) so runs from
the app are purely local afterwards. First run of everything is network-bound
and can take an hour or more in total — run it once, ideally overnight:

    python scripts/warm_metros.py                 # all metros
    python scripts/warm_metros.py miami seattle   # a subset
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from welfare_rs import params  # noqa: E402
from welfare_rs.metro import build_metro_network  # noqa: E402


def main() -> None:
    metros = sys.argv[1:] or list(params.METRO_PARAMS["metros"])
    failures = []
    for metro in metros:
        label = params.METRO_PARAMS["metros"][metro]["label"]
        print(f"\n=== {metro} ({label}) ===", flush=True)
        t0 = time.time()
        try:
            net = build_metro_network(metro)
        except Exception as exc:  # keep going; report at the end
            failures.append((metro, f"{type(exc).__name__}: {exc}"))
            print(f"  FAILED: {exc}", flush=True)
            continue
        core = len(net._core_nodes or [])
        commutes = (f" · {net.num_commute_pairs:,} commute pairs"
                    if net.has_commutes else " · NO commute pairs (fallback sampling)")
        print(
            f"  {net.num_base_nodes:,} intersections ({core:,} in core) · "
            f"{net.num_edges:,} edges{commutes} · {time.time() - t0:.0f}s",
            flush=True,
        )

    if failures:
        print("\nFailed metros:")
        for metro, err in failures:
            print(f"  {metro}: {err}")
        sys.exit(1)
    print("\nAll metros warmed.")


if __name__ == "__main__":
    main()

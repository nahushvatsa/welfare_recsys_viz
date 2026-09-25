"""Audit a metro's walk and bike networks before any run relies on them.

The failure these networks are prone to is SILENT: a piece of the network is
dropped or cut off, the graph still builds and routes, and the people living
there quietly snap to the far side of a river or lose every route. (The drive
graph lost all of Lake County, Indiana that way.) So this checks, per mode:

1. **Pieces.** Every connected piece of the network, with how many core homes,
   workplaces and POIs snap into it and where it is. A small piece holding
   people is either a real island (Roosevelt Island, Treasure Island) or a
   defect to fix.
2. **Snap distance.** How far core homes, workplaces and POIs move to reach the
   network. Large gaps mean a missing piece.
3. **Reachability.** For random home -> POI pairs: share unreachable, and the
   ratio of network to straight-line distance (a detour factor well above ~1.6
   signals a broken link).
4. **Consistency with the drive graph.** Walking distance should rarely exceed
   driving distance on the same pair; a large share that does means missing
   footpaths.

    python scripts/verify_active_networks.py miami [n_pairs]
"""

from __future__ import annotations

import os
import random
import sys
from collections import Counter

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)


def _load_env() -> None:
    """Same datasource bootstrap as scripts/warm_metros.py."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    for name in (".env.server", ".env"):
        path = os.path.join(_REPO, name)
        if os.path.exists(path):
            load_dotenv(path, override=False)
    if os.environ.get("WELFARE_RS_PG_DSN") and not os.environ.get("WELFARE_RS_DATASOURCE"):
        os.environ["WELFARE_RS_DATASOURCE"] = "postgres"


_load_env()

import networkx as nx  # noqa: E402

from welfare_rs.metro import build_metro_network  # noqa: E402
from welfare_rs.utils import haversine_km  # noqa: E402


def _pct(values, qs=(50, 90, 99, 100)):
    if len(values) == 0:
        return "n/a"
    arr = np.asarray(values, dtype=np.float64)
    return "  ".join(f"p{q}={np.percentile(arr, q):.3f}" for q in qs)


def main() -> None:
    metro = sys.argv[1] if len(sys.argv) > 1 else "miami"
    n_pairs = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
    drive = build_metro_network(metro)
    rng = random.Random(7)

    homes = [p for p in drive.home_block_latlon]
    pois = list((drive.poi_latlon or {}).values())
    works = []
    if drive.has_commutes:
        c = drive._commutes
        wl = np.column_stack([c.w_lat, c.w_lon])
        works = [tuple(x) for x in np.unique(wl, axis=0)]
        works = [w for w in works if drive.in_core(*w)]
    print(f"\n=== {metro}: {len(homes):,} core home blocks, {len(works):,} work blocks, "
          f"{len(pois):,} POIs ===")

    failures = []
    for mode, net in drive.mode_networks.items():
        print(f"\n--- {mode}: {net.num_nodes:,} nodes, {net.num_edges:,} edges, "
              f"matrix {'attached' if net.has_routing_matrix else 'NOT attached'}")
        comp_of = {}
        comps = sorted(nx.weakly_connected_components(net.G), key=len, reverse=True)
        for i, comp in enumerate(comps):
            for n in comp:
                comp_of[n] = i

        def census(points, label):
            count = Counter()
            where = {}
            gaps = []
            for p in points:
                node = net.nearest_node(*p)
                i = comp_of[node]
                count[i] += 1
                where.setdefault(i, p)
                gaps.append(haversine_km(p, net.node_latlon(node)))
            print(f"  {label:6} snap km  {_pct(gaps)}   >0.2 km: "
                  f"{sum(g > 0.2 for g in gaps):,} of {len(gaps):,}")
            return count, where

        h_count, h_where = census(homes, "homes")
        w_count, w_where = census(works, "works")
        p_count, p_where = census(pois, "POIs")
        print(f"  pieces: {len(comps)}")
        for i, comp in enumerate(comps):
            if i and not (h_count[i] or w_count[i] or p_count[i]):
                continue
            where = h_where.get(i) or w_where.get(i) or p_where.get(i)
            loc = f" near {where[0]:.4f},{where[1]:.4f}" if where and i else ""
            print(f"    piece {i}: {len(comp):>7,} nodes  homes {h_count[i]:>5,}  "
                  f"works {w_count[i]:>5,}  POIs {p_count[i]:>5,}{loc}")
        # A core can legitimately be several large pieces (San Francisco and
        # Oakland share no walkable link), so what must be rare is homes on
        # SMALL pieces: under 5% of the largest piece's size.
        big = len(comps[0])
        stranded = sum(c for i, c in h_count.items() if len(comps[i]) < 0.05 * big)
        if stranded / max(1, len(homes)) > 0.02:
            failures.append(f"{mode}: {stranded / len(homes):.1%} of core homes on small pieces")

        # Reachability and detour, home -> POI. Unreachability is judged on
        # SHORT pairs (under 2 km apart), where no bay or river crossing can
        # excuse it; across a whole two-city core most pairs are legitimately
        # unreachable.
        unreachable = short = short_unreachable = 0
        detour, walk_over_drive = [], []
        for _ in range(n_pairs):
            a, b = rng.choice(homes), rng.choice(pois)
            km = net.route_km_or_none(a, b)
            is_short = haversine_km(a, b) < 2.0
            short += is_short
            if km is None:
                unreachable += 1
                short_unreachable += is_short
                continue
            sl = haversine_km(a, b)
            if sl > 0.3:
                detour.append(km / sl)
            dk = drive.route_length_km_latlon(a, b)
            if dk > 0.3:
                walk_over_drive.append(km / dk)
        print(f"  home->POI unreachable: {unreachable:,} of {n_pairs:,} "
              f"({unreachable / n_pairs:.2%}); under 2 km apart: {short_unreachable:,} "
              f"of {short:,}")
        print(f"  detour (network / straight line)  {_pct(detour)}")
        print(f"  {mode} km / drive km              {_pct(walk_over_drive)}")
        if short and short_unreachable / short > 0.02:
            failures.append(f"{mode}: {short_unreachable / short:.1%} of home->POI pairs "
                            "under 2 km apart unreachable")

    print()
    if failures:
        print("CHECK:", *failures, sep="\n  ")
    else:
        print("OK")


if __name__ == "__main__":
    main()

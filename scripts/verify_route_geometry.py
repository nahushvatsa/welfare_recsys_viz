"""Verify scipy-reconstructed route polylines against the osmnx implementation.

The correctness standard is **equal path cost**, not an identical node list.
Both routers return *a* shortest path; where several tie, they may pick
different ones. So this asserts the total routed weight matches exactly, and
separately reports how often the node sequence differs and whether the drawn
polyline's length changes.

    python scripts/verify_route_geometry.py nyc [n_routes]
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)


def _load_env() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    for name in (".env.server", ".env"):
        p = os.path.join(_REPO, name)
        if os.path.exists(p):
            load_dotenv(p, override=False)
    if os.environ.get("WELFARE_RS_PG_DSN") and not os.environ.get("WELFARE_RS_DATASOURCE"):
        os.environ["WELFARE_RS_DATASOURCE"] = "postgres"


_load_env()

import osmnx as ox  # noqa: E402

from welfare_rs import params  # noqa: E402
from welfare_rs.metro import build_metro_network  # noqa: E402
from welfare_rs.utils import haversine_km  # noqa: E402


def path_weight(G, route, weight) -> float:
    total = 0.0
    for u, v in zip(route[:-1], route[1:]):
        total += min(float(d.get(weight, 1.0)) for d in G[u][v].values())
    return total


def polyline_km(pts) -> float:
    return sum(haversine_km(a, b) for a, b in zip(pts[:-1], pts[1:]))


def main() -> None:
    metro = sys.argv[1] if len(sys.argv) > 1 else "nyc"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 60

    net = build_metro_network(metro, cache_dir=params.GEO_PARAMS["cache_dir"])
    net._ensure_csr()
    weight = net._route_weight()

    rng = np.random.default_rng(4242)
    U = net._mat_nodes if net._mat_nodes is not None else np.array(net._base_nodes)
    pairs = [(int(U[a]), int(U[b]))
             for a, b in zip(rng.choice(len(U), n * 2), rng.choice(len(U), n * 2))]
    pairs = [(a, b) for a, b in pairs if a != b][:n]

    worst_w = 0.0
    diff_nodes = 0
    worst_km = 0.0
    compared = 0
    t_new = t_old = 0.0

    for orig, dest in pairs:
        t0 = time.time()
        new_route = net._route_nodes(orig, dest)
        new_pts = net.route_geometry_latlon(net.node_latlon(orig), net.node_latlon(dest))
        t_new += time.time() - t0

        t0 = time.time()
        old_route = ox.routing.shortest_path(net.G, orig, dest, weight=weight)
        t_old += time.time() - t0
        if not old_route or not new_route:
            continue

        w_new = path_weight(net.G, new_route, weight)
        w_old = path_weight(net.G, old_route, weight)
        worst_w = max(worst_w, abs(w_new - w_old))
        if new_route != old_route:
            diff_nodes += 1

        old_pts = []
        for u, v in zip(old_route[:-1], old_route[1:]):
            for ll in net._edge_points(u, v, weight):
                if not old_pts or old_pts[-1] != ll:
                    old_pts.append(ll)
        if old_pts and new_pts:
            worst_km = max(worst_km, abs(polyline_km(new_pts) - polyline_km(old_pts)))
        compared += 1

    print(f"{metro}: {compared} routes ({net.num_nodes:,} node graph, weight={weight})")
    print(f"  max path-cost difference : {worst_w:.6e}  ({'EXACT' if worst_w == 0 else 'MISMATCH'})")
    print(f"  different node sequence  : {diff_nodes}/{compared} (equal-cost ties)")
    print(f"  max polyline length diff : {worst_km:.6f} km")
    print(f"  scipy path + geometry    : {t_new:6.2f}s ({t_new / max(compared,1) * 1000:6.1f} ms/route)")
    print(f"  osmnx shortest_path only : {t_old:6.2f}s ({t_old / max(compared,1) * 1000:6.1f} ms/route)")
    print(f"  speedup                  : {t_old / max(t_new, 1e-9):.1f}x")

    if worst_w != 0.0:
        print("  FAIL: reconstructed path is not optimal")
        sys.exit(1)
    print("  PASS: every reconstructed path has identical routed cost")


if __name__ == "__main__":
    main()

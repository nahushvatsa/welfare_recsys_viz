"""Verify a metro's precomputed routing matrices against live Dijkstra.

Two independent checks, because they catch different classes of mistake:

1. **Storage** — every stored cell must equal a freshly computed Dijkstra to the
   bit. This is where an indexing bug shows up: a mis-assigned worker slice, a
   universe/row mismatch, a transposed lookup. Compared exactly (both sides are
   float64 produced by the same code path), so any difference at all is a bug.

2. **Wiring** — ``route_length_km`` / ``route_car_latlon`` must return the same
   answers with the matrix attached as with it detached. This is where a wrong
   unit conversion or a mishandled unreachable pair shows up.

3. **Coverage** — a real simulation must issue *zero* live Dijkstras. This is
   where a gap in the endpoint universe shows up: the run stays correct (the
   lazy path is still there) but silently pays for a shortest-path search, which
   is the entire cost this cache exists to remove.

    python scripts/verify_routing_matrix.py miami [n_sources] [n_targets]
"""

from __future__ import annotations

import os
import sys

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)


def _load_env() -> None:
    """Same datasource bootstrap as scripts/warm_metros.py — verifying against
    LocalDataSource would compare the matrices to a different POI catalog."""
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

from welfare_rs import params, routing_matrix  # noqa: E402
from welfare_rs.geo import fastest_path_meters  # noqa: E402
from welfare_rs.metro import build_metro_network  # noqa: E402


def main() -> None:
    metro = sys.argv[1] if len(sys.argv) > 1 else "miami"
    n_src = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    n_dst = int(sys.argv[3]) if len(sys.argv) > 3 else 500
    cache = params.GEO_PARAMS["cache_dir"]

    net = build_metro_network(metro, cache_dir=cache)
    if not net.has_routing_matrix:
        print(f"{metro}: NO routing matrix attached — nothing to verify")
        sys.exit(1)

    meta = routing_matrix.read_meta(cache, metro) or {}
    universe = net._mat_nodes
    u = universe.size
    print(f"{metro}: universe {u:,}  graph {net.num_nodes:,} nodes  "
          f"built {meta.get('built_at', '?')}")

    net._ensure_csr()
    net._ensure_time_edge_index()
    from scipy.sparse.csgraph import dijkstra

    rng = np.random.default_rng(987654321)
    src_idx = rng.choice(u, size=min(n_src, u), replace=False)
    dst_idx = rng.choice(u, size=min(n_dst, u), replace=False)
    dst_rows = np.array([net._csr_row[int(universe[j])] for j in dst_idx])

    # ── check 1: stored cells == live Dijkstra, exactly ──────────────────────
    bad = 0
    checked = 0
    for i in src_idx:
        src_node = int(universe[i])
        src_row = net._csr_row[src_node]

        d = dijkstra(net._csr, directed=True, indices=src_row)
        if not np.array_equal(net._mat_dist_m[i, dst_idx], d[dst_rows],
                              equal_nan=True):
            bad += 1
            print(f"  MISMATCH dist_m at source index {i} (node {src_node})")

        if net._mat_car_sec is not None:
            sec, preds = dijkstra(net._csr_t, directed=True, indices=src_row,
                                  return_predecessors=True)
            m = fastest_path_meters(net._tlen_keys, net._tlen_vals, sec, preds)
            m = np.where(np.isfinite(sec), m, np.inf)
            if not np.array_equal(net._mat_car_sec[i, dst_idx], sec[dst_rows],
                                  equal_nan=True):
                bad += 1
                print(f"  MISMATCH car_sec at source index {i} (node {src_node})")
            if not np.array_equal(net._mat_car_m[i, dst_idx], m[dst_rows],
                                  equal_nan=True):
                bad += 1
                print(f"  MISMATCH car_m at source index {i} (node {src_node})")
        checked += len(dst_idx)

    print(f"  storage : {checked:,} pairs x 3 tables — "
          f"{'FAIL (%d mismatched rows)' % bad if bad else 'exact match'}")

    # ── check 2: public API agrees with and without the matrix ───────────────
    pair_src = rng.choice(u, size=min(150, u), replace=False)
    pair_dst = rng.choice(u, size=min(150, u), replace=False)
    pairs = [(int(universe[a]), int(universe[b]))
             for a, b in zip(pair_src, pair_dst) if universe[a] != universe[b]]

    with_km = [net.route_length_km(a, b) for a, b in pairs]
    with_car = [net.route_car_latlon(net.node_latlon(a), net.node_latlon(b))
                for a, b in pairs]

    # Detach and clear every cache, so the lazy path really recomputes.
    saved = (net._mat_nodes, net._mat_dist_m, net._mat_car_sec, net._mat_car_m)
    net._mat_nodes = net._mat_dist_m = net._mat_car_sec = net._mat_car_m = None
    net._dist_cache.clear(); net._car_cache.clear()
    net._ss_cache.clear(); net._ss_time_cache.clear()
    net.register_route_targets([net.node_latlon(b) for _a, b in pairs])

    lazy_km = [net.route_length_km(a, b) for a, b in pairs]
    lazy_car = [net.route_car_latlon(net.node_latlon(a), net.node_latlon(b))
                for a, b in pairs]
    net._mat_nodes, net._mat_dist_m, net._mat_car_sec, net._mat_car_m = saved

    km_bad = sum(1 for x, y in zip(with_km, lazy_km) if x != y)
    car_bad = 0
    for x, y in zip(with_car, lazy_car):
        if (x is None) != (y is None):
            car_bad += 1
        elif x is not None and (x[0] != y[0] or x[1] != y[1]):
            car_bad += 1

    print(f"  wiring  : {len(pairs)} pairs — route_length_km "
          f"{'FAIL (%d differ)' % km_bad if km_bad else 'exact match'}, "
          f"route_car_latlon "
          f"{'FAIL (%d differ)' % car_bad if car_bad else 'exact match'}")

    # ── check 3: a real run must never fall through to a live Dijkstra ───────
    from welfare_rs.datasource import get_datasource
    from welfare_rs.simulation import Simulation

    params.SIMPLIFICATION_TOGGLES["CAR_ONLY_MODE"] = True
    ds = get_datasource()
    fresh = build_metro_network(metro, cache_dir=cache)  # untouched caches
    sim = Simulation(
        num_agents=400, seed=42, use_recommenders=True,
        persona_csv_path=ds.persona_csv_path(),
        road_network=fresh, poi_rows=ds.poi_rows(metro),
        disabled_modes=("transit",),
    )
    sim.run_days(2)
    leaked_dist = len(fresh._ss_cache)
    leaked_time = len(fresh._ss_time_cache)
    off_universe = len(fresh._route_targets)
    print(f"  coverage: 400 agents x 2 days — live Dijkstras "
          f"dist={leaked_dist} time={leaked_time}, off-universe endpoints={off_universe}"
          f"{'' if not (leaked_dist or leaked_time) else '  <-- LEAK'}")

    if bad or km_bad or car_bad or leaked_dist or leaked_time:
        sys.exit(1)
    print("  PASS")


if __name__ == "__main__":
    main()

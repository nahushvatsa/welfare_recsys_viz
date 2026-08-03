"""Verify the vectorised fastest-path km accumulation against the original walk.

The old implementation summed each target's predecessor chain separately, from
the target back up to the source. The vectorised one sums the same edges by path
doubling, i.e. in a different association order, so results agree to within
floating-point round-off rather than bit-for-bit. This script reports the worst
observed deviation so that claim is measured, not assumed.

    python scripts/verify_retrace.py dc [n_sources]
"""

from __future__ import annotations

import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from welfare_rs import params  # noqa: E402


def walk_meters(net, src_row, seconds, preds, rows):
    """The original per-target predecessor walk (reference implementation)."""
    edge_len = net._time_edge_len
    out = np.full(len(rows), np.nan, dtype=np.float64)
    for i, row in enumerate(rows):
        if not np.isfinite(seconds[row]):
            continue
        meters, r = 0.0, int(row)
        while r != src_row:
            p = int(preds[r])
            if p < 0:
                meters = -1.0
                break
            meters += edge_len[(p, r)]
            r = p
        if meters >= 0.0:
            out[i] = meters
    return out


def main() -> None:
    metro = sys.argv[1] if len(sys.argv) > 1 else "dc"
    n_src = int(sys.argv[2]) if len(sys.argv) > 2 else 25
    cache = params.GEO_PARAMS["cache_dir"]

    with open(os.path.join(cache, "warmed", f"{metro}_metro.pkl"), "rb") as f:
        net = pickle.load(f)
    net._ensure_csr()
    if net._csr_t is None:
        print(f"{metro}: no travel_time on edges — nothing to verify")
        return

    from scipy.sparse.csgraph import dijkstra

    n = len(net._csr_nodes)
    rng = np.random.default_rng(12345)
    src_rows = rng.choice(n, size=min(n_src, n), replace=False)
    # Compare over a large random target sample per source (all of them would
    # make the reference walk the bottleneck, which is the whole point).
    targets = rng.choice(n, size=min(3000, n), replace=False)

    worst_abs = 0.0
    worst_rel = 0.0
    checked = 0
    t_walk = t_vec = 0.0

    for src_row in src_rows:
        seconds, preds = dijkstra(
            net._csr_t, directed=True, indices=int(src_row), return_predecessors=True
        )
        t0 = time.time()
        ref = walk_meters(net, int(src_row), seconds, preds, targets)
        t_walk += time.time() - t0

        t0 = time.time()
        vec_all = net.fastest_path_meters(int(src_row), seconds, preds)
        t_vec += time.time() - t0
        vec = vec_all[targets]

        good = ~np.isnan(ref)
        if not good.any():
            continue
        diff = np.abs(vec[good] - ref[good])
        denom = np.maximum(np.abs(ref[good]), 1.0)
        worst_abs = max(worst_abs, float(diff.max()))
        worst_rel = max(worst_rel, float((diff / denom).max()))
        checked += int(good.sum())

    print(f"{metro}: {len(src_rows)} sources x {len(targets)} targets  "
          f"({checked:,} reachable pairs compared)")
    print(f"  max absolute difference : {worst_abs:.6e} m")
    print(f"  max relative difference : {worst_rel:.6e}")
    print(f"  reference walk time     : {t_walk:.2f}s  ({t_walk / len(src_rows) * 1000:.1f} ms/source"
          f" for {len(targets)} targets)")
    print(f"  vectorised time         : {t_vec:.2f}s  ({t_vec / len(src_rows) * 1000:.1f} ms/source"
          f" for ALL {n:,} nodes)")

    tol = 1e-9
    if worst_rel > tol:
        print(f"  FAIL: relative difference exceeds {tol:g}")
        sys.exit(1)
    print(f"  PASS: agrees within {tol:g} relative")


if __name__ == "__main__":
    main()

"""Precomputed all-pairs routing over a metro's *endpoint universe*.

Why this exists
---------------
Every routing query the model makes starts and ends at one of a small, closed
set of nodes:

* agent homes and workplaces — drawn by :meth:`RoadNetwork.sample_commute`, which
  snaps LODES block points onto the graph, and
* catalog POIs — inserted into the graph as mid-block nodes at warm time.

That set is fixed by the metro's geography and data, **not** by the experiment:
1.8 M NYC commute pairs collapse onto ~15 k distinct nodes, and running 10 000
agents instead of 80 does not add a single one. So the shortest-path answers can
be computed once per metro and read back forever after.

What is stored
--------------
``<cache>/warmed/<metro>_routing/``

===================  ====================================================
``nodes.npy``        sorted int64 node ids — the endpoint universe, U
``dist_m.npy``       (U, U) float64, shortest-path metres
``car_sec.npy``      (U, U) float64, fastest-path seconds
``car_m.npy``        (U, U) float64, metres along that fastest path
``kernel.npz``       compact CSR kernel, so workers skip the graph
``meta.json``        fingerprint + shape, validated on every load
===================  ====================================================

Three tables because the fastest route is not the shortest route. ``car_sec``
times the drive; ``car_m`` measures the road actually driven, which is what fuel
cost and emissions accrue over; ``dist_m`` is the separate shortest-path metric
the day-planner scores candidate destinations with.

float64, not float32: it halves nothing that matters on a 19 TB disk and it lets
the verifier assert *exact* equality against a live Dijkstra rather than
equality-within-a-tolerance.

Rows are disjoint per worker, so the parallel fill needs no locking — each
process writes its own slice of the same ``MAP_SHARED`` memmap.
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import numpy as np

MATRIX_VERSION = 2
DTYPE = np.float64

# Sources per scipy call. Batching amortises Python overhead; the cap keeps a
# worker's scratch arrays to a few tens of MB even on the largest metro.
_BATCH = 32

_TABLES = ("dist_m", "car_sec", "car_m")


# ── paths / identity ─────────────────────────────────────────────────────────

def matrix_dir(cache_dir: str, metro: str) -> str:
    return os.path.join(cache_dir, "warmed", f"{metro}_routing")


def _table_path(mdir: str, name: str) -> str:
    return os.path.join(mdir, f"{name}.npy")


def network_fingerprint(net, poi_fingerprint: str = "") -> str:
    """Identity of the graph the matrices were built against.

    Node ids are the matrix indices, so anything that could renumber them —
    a re-download, a different POI cap, a changed core polygon — has to
    invalidate the cache. Hashing the sorted node ids catches all of it
    directly, rather than trying to enumerate the causes.
    """
    node_ids = np.sort(np.fromiter(net.G.nodes, dtype=np.int64, count=net.G.number_of_nodes()))
    h = hashlib.sha256()
    h.update(f"v{MATRIX_VERSION}|{net.G.number_of_nodes()}|{net.G.number_of_edges()}|".encode())
    h.update(poi_fingerprint.encode())
    h.update(node_ids.tobytes())
    return h.hexdigest()[:32]


# ── the endpoint universe ────────────────────────────────────────────────────

def _bulk_snap(net, lat, lon) -> set:
    """Node ids that ``(lat, lon)`` pairs snap to, via the same BallTree
    :meth:`RoadNetwork.nearest_node` queries — so the result is exactly what the
    model will resolve at run time."""
    if lat is None or len(lat) == 0:
        return set()
    net._ensure_kdtree()
    coords = np.radians(np.column_stack([np.asarray(lat, dtype=np.float64),
                                         np.asarray(lon, dtype=np.float64)]))
    idx = net._kdtree.query(coords, k=1, return_distance=False)[:, 0]
    return set(int(x) for x in np.unique(net._kdtree_nodes[idx]))


def endpoint_universe(net) -> np.ndarray:
    """Sorted int64 node ids that routing can ever start or end at.

    Three contributors, all resolved the way the model resolves them:

    * every node the metro's commute pairs snap onto (agent homes and workplaces),
    * the inserted POI nodes, and
    * whatever each POI's **true** coordinate snaps to.

    That last one is not redundant. A POI is inserted at its projection onto the
    nearest *edge*, but ``Place.location`` keeps the dataset's exact coordinate,
    and routing snaps *that*. When an intersection happens to sit closer to the
    real building than its own street projection does, the POI routes from the
    intersection instead — a couple of dozen POIs per metro. Omitting them
    silently pushed those onto the lazy path.

    Anything still outside this set (an unlayered network, a metro with no
    commute data, the rare random-intersection fallback) routes correctly
    regardless — it just falls through to the lazy per-source Dijkstra.
    """
    nodes = set(int(n) for n in net._poi_nodes.values())

    poi_latlon = getattr(net, "poi_latlon", None)
    if poi_latlon:
        coords = np.array(list(poi_latlon.values()), dtype=np.float64)
        nodes |= _bulk_snap(net, coords[:, 0], coords[:, 1])

    if net.has_commutes:
        c = net._commutes
        nodes |= _bulk_snap(net, c.h_lat, c.h_lon)
        nodes |= _bulk_snap(net, c.w_lat, c.w_lon)

    if not nodes:
        return np.empty(0, dtype=np.int64)
    return np.array(sorted(nodes), dtype=np.int64)


# ── worker ───────────────────────────────────────────────────────────────────

def _fill_rows(args: tuple) -> Tuple[int, int, float]:
    """Compute matrix rows ``[lo, hi)`` and write them into the memmaps.

    Runs in its own process: loads the compact CSR kernel (a few MB) rather than
    the pickled graph (tens of MB and seconds of parse time per worker).
    """
    mdir, lo, hi = args
    import scipy.sparse as sp
    from scipy.sparse.csgraph import dijkstra

    from .geo import fastest_path_meters

    t0 = time.time()
    k = np.load(os.path.join(mdir, "kernel.npz"))
    n = int(k["n_nodes"])
    csr_d = sp.csr_matrix((k["d_data"], k["d_indices"], k["d_indptr"]), shape=(n, n))
    csr_t = sp.csr_matrix((k["t_data"], k["t_indices"], k["t_indptr"]), shape=(n, n))
    tkeys, tvals = k["tkeys"], k["tvals"]
    urows = k["urows"]
    has_time = bool(k["has_time"])

    dist_mm = np.load(_table_path(mdir, "dist_m"), mmap_mode="r+")
    sec_mm = np.load(_table_path(mdir, "car_sec"), mmap_mode="r+")
    carm_mm = np.load(_table_path(mdir, "car_m"), mmap_mode="r+")

    for start in range(lo, hi, _BATCH):
        stop = min(start + _BATCH, hi)
        rows = urows[start:stop]

        d = dijkstra(csr_d, directed=True, indices=rows)
        dist_mm[start:stop, :] = d[:, urows]

        if has_time:
            sec, preds = dijkstra(csr_t, directed=True, indices=rows,
                                  return_predecessors=True)
            sec_mm[start:stop, :] = sec[:, urows]
            for j in range(stop - start):
                meters = fastest_path_meters(tkeys, tvals, sec[j], preds[j])
                # Unreachable targets carry a meaningless partial sum; make the
                # stored value inf so a reader cannot mistake it for a route.
                meters = np.where(np.isfinite(sec[j]), meters, np.inf)
                carm_mm[start + j, :] = meters[urows]

    dist_mm.flush(); sec_mm.flush(); carm_mm.flush()
    del dist_mm, sec_mm, carm_mm
    return lo, hi, time.time() - t0


# ── build ────────────────────────────────────────────────────────────────────

def _write_kernel(net, mdir: str, universe: np.ndarray) -> bool:
    """Persist the CSR structures workers need, without the graph."""
    net._ensure_csr()
    net._ensure_time_edge_index()
    has_time = net._csr_t is not None

    row_of = net._csr_row
    urows = np.array([row_of[int(n)] for n in universe], dtype=np.int32)

    empty32 = np.empty(0, dtype=np.int32)
    empty64 = np.empty(0, dtype=np.float64)
    np.savez(
        os.path.join(mdir, "kernel.npz"),
        n_nodes=np.int64(len(net._csr_nodes)),
        d_data=net._csr.data, d_indices=net._csr.indices, d_indptr=net._csr.indptr,
        t_data=net._csr_t.data if has_time else empty64,
        t_indices=net._csr_t.indices if has_time else empty32,
        t_indptr=net._csr_t.indptr if has_time else np.zeros(1, dtype=np.int32),
        tkeys=net._tlen_keys, tvals=net._tlen_vals,
        urows=urows, has_time=np.bool_(has_time),
    )
    return has_time


def build(net, metro: str, *, cache_dir: str, poi_fingerprint: str = "",
          workers: Optional[int] = None, log=print) -> Optional[dict]:
    """Compute and store the routing matrices for ``net``. Returns its meta dict.

    ``net`` must already be final: POIs inserted and commute pairs attached.
    Returns None when the metro has no endpoint universe (nothing to cache).
    """
    universe = endpoint_universe(net)
    if universe.size == 0:
        log(f"  {metro}: no endpoint universe (no POIs, no commutes) — skipping matrices")
        return None

    u = int(universe.size)
    mdir = matrix_dir(cache_dir, metro)
    os.makedirs(mdir, exist_ok=True)

    gb = u * u * 8 / 1e9
    log(f"  {metro}: universe {u:,} nodes -> 3 x {gb:.2f} GB ({gb * 3:.2f} GB total)")

    np.save(os.path.join(mdir, "nodes.npy"), universe)
    t0 = time.time()
    has_time = _write_kernel(net, mdir, universe)
    log(f"  {metro}: CSR kernel written in {time.time() - t0:.1f}s (time-weighted: {has_time})")

    for name in _TABLES:
        mm = np.lib.format.open_memmap(_table_path(mdir, name), mode="w+",
                                       dtype=DTYPE, shape=(u, u))
        del mm

    workers = workers or min(32, os.cpu_count() or 1)
    # Chunks smaller than one per worker so a slow slice cannot stall the tail.
    n_chunks = max(workers * 4, 1)
    bounds = np.linspace(0, u, n_chunks + 1).astype(int)
    tasks = [(mdir, int(a), int(b)) for a, b in zip(bounds[:-1], bounds[1:]) if b > a]

    t0 = time.time()
    done = 0
    # 'spawn': build() may be reached from the web service (an unwarmed metro),
    # and forking a process that already has threads running is unsafe.
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
        futs = [pool.submit(_fill_rows, t) for t in tasks]
        for fut in as_completed(futs):
            lo, hi, secs = fut.result()
            done += hi - lo
            elapsed = time.time() - t0
            rate = done / max(elapsed, 1e-9)
            eta = (u - done) / max(rate, 1e-9)
            log(f"  {metro}: {done:,}/{u:,} rows  ({elapsed:.0f}s elapsed, "
                f"~{eta:.0f}s left, {rate:.0f} rows/s)")
    build_secs = time.time() - t0

    meta = {
        "version": MATRIX_VERSION,
        "metro": metro,
        "fingerprint": network_fingerprint(net, poi_fingerprint),
        "poi_fingerprint": poi_fingerprint,
        "universe": u,
        "graph_nodes": net.G.number_of_nodes(),
        "graph_edges": net.G.number_of_edges(),
        "dtype": np.dtype(DTYPE).name,
        "has_time": bool(has_time),
        "build_seconds": round(build_secs, 1),
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(os.path.join(mdir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    log(f"  {metro}: matrices built in {build_secs:.0f}s ({u:,} sources)")
    return meta


# ── load ─────────────────────────────────────────────────────────────────────

def read_meta(cache_dir: str, metro: str) -> Optional[dict]:
    path = os.path.join(matrix_dir(cache_dir, metro), "meta.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def attach(net, metro: str, cache_dir: str, poi_fingerprint: str = "") -> bool:
    """Memory-map the matrices onto ``net``. Returns True when attached.

    Read-only maps: every worker process maps the same file, so the OS page
    cache holds exactly one copy no matter how many simulations are running.

    A fingerprint mismatch is treated as "no cache" rather than an error — the
    caller keeps the lazy Dijkstra path and stays correct, just slower.
    """
    meta = read_meta(cache_dir, metro)
    if meta is None or meta.get("version") != MATRIX_VERSION:
        return False
    if meta.get("fingerprint") != network_fingerprint(net, poi_fingerprint):
        return False

    mdir = matrix_dir(cache_dir, metro)
    try:
        nodes = np.load(os.path.join(mdir, "nodes.npy"))
        dist = np.load(_table_path(mdir, "dist_m"), mmap_mode="r")
        sec = np.load(_table_path(mdir, "car_sec"), mmap_mode="r")
        carm = np.load(_table_path(mdir, "car_m"), mmap_mode="r")
    except (OSError, ValueError):
        return False

    u = int(meta["universe"])
    if nodes.shape != (u,) or dist.shape != (u, u):
        return False

    net._mat_nodes = nodes
    net._mat_dist_m = dist
    net._mat_car_sec = sec if meta.get("has_time") else None
    net._mat_car_m = carm if meta.get("has_time") else None
    return True

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

Walk and bike networks (:mod:`welfare_rs.active_modes`) get their own set in
``<cache>/warmed/<metro>_<mode>_routing/`` holding ``dist_m`` only: they route
by length, and time is distance over the mode's speed. Their universe is the
core subset of the same endpoints, because only core residents walk or cycle.

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

def matrix_dir(cache_dir: str, metro: str, mode: str = "drive") -> str:
    # The drive tables keep their original directory, so adding modes did not
    # invalidate a single existing car matrix.
    if mode == "drive":
        return os.path.join(cache_dir, "warmed", f"{metro}_routing")
    return os.path.join(cache_dir, "warmed", f"{metro}_{mode}_routing")


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


def endpoint_universe(net, source=None, core_only: bool = False) -> np.ndarray:
    """Sorted int64 node ids that routing can ever start or end at.

    ``source`` is the network that carries the commutes, home blocks and POI
    coordinates — ``net`` itself for the drive graph, the drive graph for a
    walk or bike network, which is snapped against but holds none of them.
    ``core_only`` keeps only points inside the core polygon: walking and
    cycling are offered to core residents only, so a suburban home never
    routes on those networks and would only inflate the matrix.

    Four contributors, all resolved the way the model resolves them:

    * every node the metro's commute pairs snap onto (agent homes and workplaces),
    * every node a populated CORE block snaps onto — where a built population
      puts its non-workers, who have no commute pair to be covered by,
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
    src = source if source is not None else net

    def snap(lat, lon) -> set:
        lat = np.asarray(lat, dtype=np.float64)
        lon = np.asarray(lon, dtype=np.float64)
        if core_only and lat.size:
            keep = src.in_core_mask(lat, lon)
            lat, lon = lat[keep], lon[keep]
        return _bulk_snap(net, lat, lon)

    nodes = set(int(n) for n in net._poi_nodes.values())

    poi_latlon = getattr(src, "poi_latlon", None)
    if poi_latlon:
        coords = np.array(list(poi_latlon.values()), dtype=np.float64)
        nodes |= snap(coords[:, 0], coords[:, 1])

    home_blocks = getattr(src, "home_block_latlon", None)
    if home_blocks:
        hb = np.asarray(home_blocks, dtype=np.float64)
        nodes |= snap(hb[:, 0], hb[:, 1])

    if src.has_commutes:
        c = src._commutes
        nodes |= snap(c.h_lat, c.h_lon)
        nodes |= snap(c.w_lat, c.w_lon)

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
    tkeys, tvals = k["tkeys"], k["tvals"]
    urows = k["urows"]
    has_time = bool(k["has_time"])
    # An untimed (walk/bike) kernel stores an empty time matrix.
    csr_t = (sp.csr_matrix((k["t_data"], k["t_indices"], k["t_indptr"]), shape=(n, n))
             if has_time else None)

    dist_mm = np.load(_table_path(mdir, "dist_m"), mmap_mode="r+")
    # Untimed (walk/bike) sets have no car tables at all.
    sec_mm = np.load(_table_path(mdir, "car_sec"), mmap_mode="r+") if has_time else None
    carm_mm = np.load(_table_path(mdir, "car_m"), mmap_mode="r+") if has_time else None

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

    for mm in (dist_mm, sec_mm, carm_mm):
        if mm is not None:
            mm.flush()
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
          workers: Optional[int] = None, log=print, mode: str = "drive",
          source=None) -> Optional[dict]:
    """Compute and store the routing matrices for ``net``. Returns its meta dict.

    ``net`` must already be final: POIs inserted and commute pairs attached.
    Returns None when the metro has no endpoint universe (nothing to cache).

    For a walk or bike network pass ``mode`` and ``source`` (the drive graph,
    which carries the commutes and home blocks); the universe is then the core
    subset of the endpoints and only ``dist_m`` is stored.
    """
    active = mode != "drive"
    universe = endpoint_universe(net, source=source, core_only=active)
    tag = metro if not active else f"{metro}/{mode}"
    if universe.size == 0:
        log(f"  {tag}: no endpoint universe (no POIs, no commutes) — skipping matrices")
        return None

    u = int(universe.size)
    mdir = matrix_dir(cache_dir, metro, mode)
    os.makedirs(mdir, exist_ok=True)
    # A stale meta.json must not vouch for half-written tables if this build
    # dies part-way; it is rewritten only once every row is filled.
    meta_path = os.path.join(mdir, "meta.json")
    if os.path.exists(meta_path):
        os.remove(meta_path)

    np.save(os.path.join(mdir, "nodes.npy"), universe)
    t0 = time.time()
    has_time = _write_kernel(net, mdir, universe)
    tables = _TABLES if has_time else ("dist_m",)
    gb = u * u * 8 / 1e9
    log(f"  {tag}: universe {u:,} nodes -> {len(tables)} x {gb:.2f} GB "
        f"({gb * len(tables):.2f} GB total)")
    log(f"  {tag}: CSR kernel written in {time.time() - t0:.1f}s (time-weighted: {has_time})")

    for name in tables:
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
            log(f"  {tag}: {done:,}/{u:,} rows  ({elapsed:.0f}s elapsed, "
                f"~{eta:.0f}s left, {rate:.0f} rows/s)")
    build_secs = time.time() - t0

    meta = {
        "version": MATRIX_VERSION,
        "metro": metro,
        "mode": mode,
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
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    log(f"  {tag}: matrices built in {build_secs:.0f}s ({u:,} sources)")
    return meta


# ── load ─────────────────────────────────────────────────────────────────────

def read_meta(cache_dir: str, metro: str, mode: str = "drive") -> Optional[dict]:
    path = os.path.join(matrix_dir(cache_dir, metro, mode), "meta.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def attach(net, metro: str, cache_dir: str, poi_fingerprint: str = "",
           mode: str = "drive") -> bool:
    """Memory-map the matrices onto ``net``. Returns True when attached.

    Read-only maps: every worker process maps the same file, so the OS page
    cache holds exactly one copy no matter how many simulations are running.

    A fingerprint mismatch is treated as "no cache" rather than an error — the
    caller keeps the lazy Dijkstra path and stays correct, just slower.
    """
    meta = read_meta(cache_dir, metro, mode)
    if meta is None or meta.get("version") != MATRIX_VERSION:
        return False
    # Metas written before modes existed carry no key and are drive tables.
    if meta.get("mode", "drive") != mode:
        return False
    if meta.get("fingerprint") != network_fingerprint(net, poi_fingerprint):
        return False

    mdir = matrix_dir(cache_dir, metro, mode)
    try:
        nodes = np.load(os.path.join(mdir, "nodes.npy"))
        dist = np.load(_table_path(mdir, "dist_m"), mmap_mode="r")
        sec = carm = None
        if meta.get("has_time"):
            sec = np.load(_table_path(mdir, "car_sec"), mmap_mode="r")
            carm = np.load(_table_path(mdir, "car_m"), mmap_mode="r")
    except (OSError, ValueError):
        return False

    u = int(meta["universe"])
    if nodes.shape != (u,) or dist.shape != (u, u):
        return False

    net._mat_nodes = nodes
    net._mat_dist_m = dist
    net._mat_car_sec = sec
    net._mat_car_m = carm
    return True

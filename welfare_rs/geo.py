"""OSM road-network backend for the travel-behaviour ABM.

This module provides the real street network (downloaded from OpenStreetMap via
OSMnx) that the simulation runs on. It is deliberately free of any ABM
knowledge: it only snaps coordinates to network nodes, measures network routes
(length + geometry), and answers single-source distance queries used by the
agent day-planner.

Design notes
------------
* The network is downloaded **once** and cached to GraphML on disk. The
  simulation never hits the Overpass API at run time, so there is no per-run
  API budget — only local routing.
* Runtime artifacts are written under ``WELFARE_RS_CACHE``
  (default ``~/.cache/welfare_rs``) to avoid macOS TCC-protected folders such
  as ``~/Desktop``.
* Locations throughout the OSM-backed model are ``(lat, lon)`` tuples — the same
  shape the grid model used for ``(x, y)`` — so the rest of the ABM changes
  minimally. Snapping and routing are cached so repeated queries are cheap.
* Car (and the disabled transit stand-in) route on the drive network. Walking
  and cycling route on their own OSM networks — footways, paths and the correct
  one-way rules — built for the metro core by :mod:`welfare_rs.active_modes`
  and attached to the drive network as ``mode_networks``. The same class serves
  both; an active network is simply built with ``timed=False`` because car
  speeds mean nothing on a footway. Mode differences in speed, wait, cost and
  comfort live in ``params.MODE_PARAMS`` and are applied downstream.
"""

from __future__ import annotations

import os
import random
from typing import Dict, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
import osmnx as ox
from shapely import STRtree
from shapely.geometry import LineString, Point
from shapely.ops import substring

from .utils import haversine_km  # re-exported: geo is the historic import site

LatLon = Tuple[float, float]
BBox = Tuple[float, float, float, float]  # (west, south, east, north)

DEFAULT_CACHE_DIR = os.environ.get(
    "WELFARE_RS_CACHE",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache"),
)


def fastest_path_meters(tlen_keys, tlen_vals, seconds, preds):
    """Metres along the fastest path from a Dijkstra source to **every** node.

    Vectorised replacement for walking each target's predecessor chain
    separately. Two observations make it work:

    * ``preds`` is a tree rooted at the source, so every node's path length is
      its parent's plus one edge. The per-target walks re-summed the same trunk
      edges thousands of times over.
    * Those chain sums can be built by path doubling: after ``i`` rounds each
      node holds the sum over the ``2**i`` edges above it and points ``2**i``
      steps further up. ``ceil(log2(depth))`` vectorised rounds replace one
      Python walk per target.

    ``tlen_keys`` / ``tlen_vals`` encode the time-weighted graph's edges as a
    sorted ``row_u * n + row_v -> metres`` index (see
    :meth:`RoadNetwork._ensure_time_edge_index`).

    Values for unreachable nodes are meaningless — their ``seconds`` is inf — so
    callers must mask on ``np.isfinite(seconds)``, exactly as the per-target
    walk skipped them. Summation order differs from the walk, so results agree
    to floating-point round-off (measured at ~1e-15 relative) rather than
    bit-for-bit; see ``scripts/verify_retrace.py``.

    Module-level rather than a method so the matrix precompute can call it with
    a bare CSR kernel, without materialising a whole :class:`RoadNetwork`.
    """
    n = int(seconds.shape[0])
    parent = np.asarray(preds, dtype=np.int64)
    has_parent = parent >= 0

    # Length of the edge (parent[r] -> r) for every node that has a parent.
    edge_w = np.zeros(n, dtype=np.float64)
    if has_parent.any() and tlen_keys.size:
        child = np.nonzero(has_parent)[0]
        query = parent[child] * n + child
        idx = np.searchsorted(tlen_keys, query)
        np.clip(idx, 0, tlen_keys.size - 1, out=idx)
        found = tlen_keys[idx] == query
        edge_w[child] = np.where(found, tlen_vals[idx], 0.0)

    # Path doubling. Each round: add the ancestor's partial sum, then jump twice
    # as far up. A node that has reached the root stops accumulating. log2(n)
    # rounds always suffice for a tree of n nodes; the bound only guarantees
    # termination were ``preds`` ever malformed.
    total = edge_w.copy()
    anc = np.where(has_parent, parent, -1)
    for _ in range(int(n).bit_length() + 1):
        active = anc >= 0
        if not active.any():
            break
        safe = np.where(active, anc, 0)
        total = total + np.where(active, total[safe], 0.0)
        anc = np.where(active, anc[safe], -1)
    return total


class RoadNetwork:
    """A cached OSM street network with snapping and routing helpers.

    Construct from either a bounding box or a centre point + radius::

        RoadNetwork("nyc_midtown", center=(40.758, -73.9855), dist_m=2000)
        RoadNetwork("nyc_manhattan", bbox=(-74.02, 40.70, -73.93, 40.80))
    """

    def __init__(
        self,
        name: str,
        *,
        bbox: Optional[BBox] = None,
        center: Optional[LatLon] = None,
        dist_m: Optional[float] = None,
        graph=None,
        network_type: str = "drive",
        cache_dir: Optional[str] = None,
        simplify: bool = True,
        strongly_connected: bool = True,
        timed: bool = True,
    ):
        self.name = name
        self.network_type = network_type
        self.cache_dir = cache_dir or DEFAULT_CACHE_DIR
        self._networks_dir = os.path.join(self.cache_dir, "networks")
        os.makedirs(self._networks_dir, exist_ok=True)
        # We manage our own GraphML cache; disable OSMnx's Overpass cache, which
        # would otherwise try to write to a (possibly unwritable) ./cache dir.
        ox.settings.use_cache = False

        if graph is not None:
            # Pre-assembled graph (metro.py composes core + arterial shell).
            self.G = graph
        else:
            self.G = self._load_or_download(
                bbox=bbox,
                center=center,
                dist_m=dist_m,
                simplify=simplify,
                strongly_connected=strongly_connected,
            )
        # Per-edge speeds/travel times (car routing minimises time so freeway
        # commutes beat surface streets). Upgraded-in-place old caches persist.
        # An active-mode network (walk/bike) is untimed: osmnx would impute CAR
        # speeds from highway class, which say nothing about walking or cycling
        # and would make the router minimise the wrong quantity. Those networks
        # route by length and apply the mode's own speed downstream.
        if not timed:
            self._has_edge_times = False
        elif self._ensure_travel_times() and graph is None:
            ox.save_graphml(self.G, self._graph_path())

        # Node coordinate lookups + sampling pool.
        self.nodes: List[int] = list(self.G.nodes)
        self._lat: Dict[int, float] = {}
        self._lon: Dict[int, float] = {}
        for n, d in self.G.nodes(data=True):
            self._lat[n] = float(d["y"])
            self._lon[n] = float(d["x"])

        lats = list(self._lat.values())
        lons = list(self._lon.values())
        # bounds = (south, west, north, east); center used for the basemap view.
        self.bounds: BBox = (min(lats), min(lons), max(lats), max(lons))
        self.center: LatLon = center or (sum(lats) / len(lats), sum(lons) / len(lons))

        # Caches: coord->node snap, (orig,dest)->km, source->pruned distances,
        # (orig,dest)->route polyline.
        self._snap_cache: Dict[Tuple[float, float], int] = {}
        self._dist_cache: Dict[Tuple[int, int], float] = {}
        # Single-source results are pruned to registered *target* nodes (agent
        # homes/works + POIs) — storing distances to every intermediate
        # intersection serves no query and dominates memory at metro scale.
        # Entries are (targets_version, {target_node: meters}).
        self._ss_cache: Dict[int, Tuple[int, Dict[int, float]]] = {}
        # Time-weighted analogues for car routing: source -> {target: (min, km)}
        # (km measured along the *fastest* path), and a per-pair front cache.
        self._ss_time_cache: Dict[int, Tuple[int, Dict[int, Tuple[float, float]]]] = {}
        self._car_cache: Dict[Tuple[int, int], Tuple[float, float]] = {}
        self._geom_cache: Dict[Tuple[int, int], List[LatLon]] = {}
        self._route_targets: set = set()
        self._targets_version = 0

        # Lazily-built acceleration structures (invalidated by _refresh_nodes):
        #   _kdtree: haversine BallTree over node coords — O(log n) snapping,
        #            built once instead of osmnx rebuilding its index per call.
        #   _csr:    CSR adjacency (min length per directed edge) — lets scipy
        #            run Dijkstra in C. Both yield identical results to before.
        #   _csr_t:  CSR adjacency weighted by travel_time (car routing), with
        #            _time_edge_len mapping each kept edge to its length so km
        #            along the fastest path can be accumulated.
        self._kdtree = None
        self._kdtree_nodes = None  # np.ndarray: tree row -> node id
        self._csr = None
        self._csr_row: Optional[Dict[int, int]] = None   # node id -> CSR row
        self._csr_nodes = None     # np.ndarray: CSR row -> node id
        self._csr_t = None
        self._time_edge_len: Optional[Dict[Tuple[int, int], float]] = None
        # Vectorised form of _time_edge_len: edges encoded as
        # ``row_u * n + row_v`` in a sorted int64 array, with lengths alongside.
        # Lets the fastest-path km accumulation look up every predecessor edge
        # in one searchsorted instead of a per-target Python walk.
        self._tlen_keys = None
        self._tlen_vals = None

        # Precomputed routing matrices (welfare_rs.routing_matrix), attached
        # after load when the metro has been warmed. None => lazy Dijkstra only.
        self._mat_nodes = None      # sorted int64 node ids (the endpoint universe)
        self._mat_dist_m = None     # (U, U) float64 memmap: shortest-path metres
        self._mat_car_sec = None    # (U, U) float64 memmap: fastest-path seconds
        self._mat_car_m = None      # (U, U) float64 memmap: metres along the fastest path
        # Materialised matrix rows, keyed by (table tag, row). The planner scores
        # many destinations from one origin, so pulling the whole row on first
        # touch turns dozens of random page faults into one sequential read —
        # which matters because the cache lives on spinning disk until the page
        # cache has it. Bounded in _matrix_lookup; never pickled.
        self._mat_rows: Dict[Tuple[str, int], object] = {}
        # Dijkstra predecessor arrays keyed by CSR source row, for reconstructing
        # route polylines. One entry serves every destination from that origin.
        self._pred_cache: Dict[int, object] = {}

        # Original road (intersection) nodes — homes/work/organic locations sample
        # only these. POIs may be inserted as extra mid-block nodes (see
        # add_pois_as_nodes) without polluting that pool.
        self._base_nodes: List[int] = list(self.nodes)
        self._poi_nodes: Dict[str, int] = {}
        self._next_poi_node_id = 10_000_000_000_000

        # Two-layer metro state (attach_layers): the principal-city polygon,
        # its node pool, and per-county core/shell pools for home sampling.
        # None on plain single-area networks.
        self.core_polygon = None
        self.county_meta: Optional[Dict[str, dict]] = None
        self._core_nodes: Optional[List[int]] = None
        self._county_pools: Optional[Dict[str, Dict[str, List[int]]]] = None
        self._county_ids: Optional[List[str]] = None
        self._county_weights: Optional[List[float]] = None
        self._core_bounds: Optional[BBox] = None
        self._core_center: Optional[LatLon] = None

        # Real block-level commutes (attach_commutes). When present these
        # replace both provisional samplers: homes stop being uniform-within-
        # county and workplaces stop being uniform-within-core.
        self._commutes = None
        self._commute_cum = None      # prefix sum of job weights
        self._commute_total = 0.0
        self._commute_seg_cum = {}    # LODES age segment -> (prefix sum, total)

        # Populated CORE blocks (attach_home_blocks). Not a sampler: the
        # non-worker homes in a built population come from these blocks, and
        # the routing precompute has to know about them or those agents route
        # off the precomputed matrix. See routing_matrix.endpoint_universe.
        self._home_block_latlon = None

        # Metro key this network belongs to (set by metro.build_metro_network,
        # never pickled into meaning: it is re-set on every load).
        self.metro: Optional[str] = None
        # Walk/bike networks for the metro core, keyed by mode. Attached by
        # metro.build_metro_network after load and never pickled with the drive
        # graph — each has its own warm pickle and routing matrices.
        self.mode_networks: Dict[str, "RoadNetwork"] = {}
        # Pair cache for route_km_or_none: km, or None when unreachable.
        self._reach_cache: Dict[Tuple[int, int], Optional[float]] = {}

    def __getstate__(self) -> dict:
        """Exclude the lazily-rebuilt acceleration structures from pickling (so
        the disk cache stays small and isn't coupled to sklearn/scipy pickle
        versions); they rebuild on first use after a load. The graph + routing
        caches (the expensive, reusable parts) are persisted.

        Commute pairs are dropped too — a metro can carry millions of them, and
        every worker process re-attaches them from the datasource after loading
        the warm pickle (see metro.build_metro_network), the same way POI rows
        are fetched per worker rather than baked into the graph cache."""
        state = self.__dict__.copy()
        for k in ("_kdtree", "_kdtree_nodes", "_csr", "_csr_row", "_csr_nodes",
                  "_csr_t", "_time_edge_len", "_tlen_keys", "_tlen_vals",
                  "_mat_nodes", "_mat_dist_m", "_mat_car_sec", "_mat_car_m",
                  "_commutes", "_commute_cum"):
            state[k] = None
        state["_mat_rows"] = {}
        state["_pred_cache"] = {}
        # Separate warm pickles (see active_modes); re-attached on every load.
        state["mode_networks"] = {}
        state["_commute_total"] = 0.0
        # Per-segment prefix sums are as long as the pair table (three arrays of
        # a few million floats on a large metro), and are rebuilt lazily from the
        # pairs the worker re-attaches, so they never belong in the cache.
        state["_commute_seg_cum"] = {}
        return state

    # Attributes added after warm pickles were first written. Restoring them
    # explicitly means an older cache still loads instead of raising
    # AttributeError deep inside a routing call.
    _LAZY_DEFAULTS = (
        "_kdtree", "_kdtree_nodes", "_csr", "_csr_row", "_csr_nodes", "_csr_t",
        "_time_edge_len", "_tlen_keys", "_tlen_vals",
        "_mat_nodes", "_mat_dist_m", "_mat_car_sec", "_mat_car_m",
        "_commutes", "_commute_cum",
    )

    def __setstate__(self, state: dict) -> None:
        for key in self._LAZY_DEFAULTS:
            state.setdefault(key, None)
        state.setdefault("_commute_total", 0.0)
        state.setdefault("_commute_seg_cum", {})
        if state.get("_commute_seg_cum") is None:
            state["_commute_seg_cum"] = {}
        state.setdefault("_mat_rows", {})
        if state.get("_mat_rows") is None:  # pickled before the row cache existed
            state["_mat_rows"] = {}
        if state.get("_pred_cache") is None:
            state["_pred_cache"] = {}
        state.setdefault("metro", None)
        if not state.get("mode_networks"):
            state["mode_networks"] = {}
        if state.get("_reach_cache") is None:
            state["_reach_cache"] = {}
        self.__dict__.update(state)

    # ── construction ─────────────────────────────────────────────────────────

    def _graph_path(self) -> str:
        return os.path.join(self._networks_dir, f"{self.name}_{self.network_type}.graphml")

    def _load_or_download(self, *, bbox, center, dist_m, simplify, strongly_connected):
        path = self._graph_path()
        if os.path.exists(path):
            return ox.load_graphml(path)

        if bbox is not None:
            graph = ox.graph_from_bbox(bbox, network_type=self.network_type, simplify=simplify)
        elif center is not None and dist_m is not None:
            graph = ox.graph_from_point(
                center, dist=dist_m, network_type=self.network_type, simplify=simplify
            )
        else:
            raise ValueError(
                "Provide either bbox=(west,south,east,north) or center=(lat,lon) with dist_m."
            )

        # Reduce to a routable core so shortest_path never raises NoPath.
        graph = ox.truncate.largest_component(graph, strongly=strongly_connected)
        ox.save_graphml(graph, path)
        return graph

    def _ensure_travel_times(self) -> bool:
        """Guarantee per-edge ``speed_kph``/``travel_time`` attributes.

        Speeds come from OSM ``maxspeed`` where tagged, imputed by highway
        class otherwise (OSMnx's standard method) — this is what makes a
        freeway commute faster than the same km on surface streets. Returns
        True when attributes were added (caller may want to re-save). Sets
        ``self._has_edge_times`` either way; on failure (very old osmnx)
        routing falls back to distance / flat mode speeds.
        """
        edge = next(iter(self.G.edges(data=True)), None)
        if edge is not None and "travel_time" in edge[2]:
            self._has_edge_times = True
            return False
        try:
            ox.routing.add_edge_speeds(self.G)
            ox.routing.add_edge_travel_times(self.G)
            self._has_edge_times = True
            return True
        except Exception:
            self._has_edge_times = False
            return False

    # ── snapping ───────────────────────────────────────────────────────────--

    def nearest_node(self, lat: float, lon: float) -> int:
        """Snap a raw (lat, lon) to the id of the nearest network node (cached).

        Queries a prebuilt haversine ``BallTree`` (built once and reused) rather
        than ``osmnx.nearest_nodes``, which rebuilds its spatial index on every
        call. Returns the identical node id (same BallTree-haversine method osmnx
        uses for unprojected graphs); ~3 orders of magnitude faster on cache miss.
        """
        key = (round(lat, 6), round(lon, 6))
        node = self._snap_cache.get(key)
        if node is None:
            self._ensure_kdtree()
            idx = int(self._kdtree.query(np.radians([[lat, lon]]), k=1,
                                         return_distance=False)[0][0])
            node = int(self._kdtree_nodes[idx])
            self._snap_cache[key] = node
        return node

    def _ensure_kdtree(self) -> None:
        """Build (once) the haversine BallTree over current node coordinates."""
        if self._kdtree is not None:
            return
        from sklearn.neighbors import BallTree  # same backend osmnx uses

        node_ids = list(self.G.nodes)
        coords = np.radians(
            np.array([[self._lat[n], self._lon[n]] for n in node_ids], dtype=np.float64)
        )
        self._kdtree = BallTree(coords, metric="haversine")
        self._kdtree_nodes = np.array(node_ids)

    def node_latlon(self, node: int) -> LatLon:
        return (self._lat[node], self._lon[node])

    def snap_latlon(self, lat: float, lon: float) -> LatLon:
        """Return the (lat, lon) of the nearest network node to a raw point.

        Storing snapped node coordinates (rather than raw inputs) keeps every
        location in the model exactly on the graph, so routing never has to
        re-snap and the snap cache stays exact.
        """
        return self.node_latlon(self.nearest_node(lat, lon))

    # ── routing ──────────────────────────────────────────────────────────────

    def route_length_km(self, orig: int, dest: int) -> float:
        """Network shortest-path length in km between two node ids (cached).

        Served from the (cached) single-source tree out of ``orig`` rather than a
        fresh per-pair search: the planner scores many destinations from the same
        origin, so one Dijkstra amortises across all of them. Identical values.

        Destinations are registered as route targets on first use (see
        ``register_route_targets``); a query for a not-yet-registered target
        self-heals by registering it and recomputing the source tree once.
        """
        if orig == dest:
            return 0.0
        key = (orig, dest)
        dist = self._dist_cache.get(key)
        if dist is not None:
            return dist

        meters = self._matrix_lookup(self._mat_dist_m, orig, dest, "d")
        if meters is None:
            # Outside the precomputed universe (or no matrix): fall back to the
            # lazy per-source tree, registering the destination as a target.
            if dest not in self._route_targets:
                self._route_targets.add(dest)
                self._targets_version += 1
            meters = self._single_source(orig).get(dest)

        if meters is None or not np.isfinite(meters):
            # Unreachable (shouldn't happen on the routable core).
            dist = haversine_km(self.node_latlon(orig), self.node_latlon(dest))
        else:
            dist = meters / 1000.0
        self._dist_cache[key] = dist
        return dist

    def route_length_km_latlon(self, a: LatLon, b: LatLon) -> float:
        return self.route_length_km(self.nearest_node(*a), self.nearest_node(*b))

    def route_km_or_none(self, a: LatLon, b: LatLon) -> Optional[float]:
        """Shortest-path km between two points, or None when unreachable.

        :meth:`route_length_km` substitutes the straight-line distance for an
        unreachable pair, which is harmless on the strongly-connected drive
        graph but wrong on a walk or bike network: those keep genuinely
        separate components (San Francisco and Oakland, with no walkable link
        across the Bay), and a straight-line stand-in would let an agent walk
        across the water. Here unreachable means the mode is not available.
        """
        orig = self.nearest_node(*a)
        dest = self.nearest_node(*b)
        if orig == dest:
            return 0.0
        key = (orig, dest)
        if key in self._reach_cache:
            return self._reach_cache[key]
        meters = self._matrix_lookup(self._mat_dist_m, orig, dest, "d")
        if meters is None:
            if dest not in self._route_targets:
                self._route_targets.add(dest)
                self._targets_version += 1
            meters = self._single_source(orig).get(dest)
        km = None if (meters is None or not np.isfinite(meters)) else meters / 1000.0
        self._reach_cache[key] = km
        return km

    def snap_gap_km(self, lat: float, lon: float) -> float:
        """Straight-line km from a point to the node it snaps to."""
        return haversine_km((lat, lon), self.node_latlon(self.nearest_node(lat, lon)))

    def _ensure_csr(self) -> None:
        """Build (once) the CSR adjacencies: minimum ``length`` among parallel
        directed edges for each (u, v) — identical distances to networkx
        Dijkstra, but scipy runs in C — and, when every edge carries a
        ``travel_time``, a second matrix of minimum time (keeping that fastest
        edge's length, so km along fastest paths can be accumulated)."""
        if self._csr is not None:
            return
        import scipy.sparse as sp

        node_ids = list(self.G.nodes)
        row_of = {n: i for i, n in enumerate(node_ids)}
        best: Dict[Tuple[int, int], float] = {}
        best_t: Dict[Tuple[int, int], Tuple[float, float]] = {}  # (sec, meters)
        timed_all = True
        for u, v, d in self.G.edges(data=True):
            w = float(d.get("length", 1.0))
            key = (row_of[u], row_of[v])
            if key not in best or w < best[key]:
                best[key] = w
            t = d.get("travel_time")
            if t is None:
                timed_all = False
            else:
                t = float(t)
                cur = best_t.get(key)
                if cur is None or t < cur[0]:
                    best_t[key] = (t, w)
        n = len(node_ids)

        def _to_csr(weights: Dict[Tuple[int, int], float]):
            if weights:
                keys = list(weights.keys())
                rows = np.fromiter((k[0] for k in keys), dtype=np.int32, count=len(keys))
                cols = np.fromiter((k[1] for k in keys), dtype=np.int32, count=len(keys))
                data = np.fromiter((weights[k] for k in keys), dtype=np.float64, count=len(keys))
            else:
                rows = cols = np.empty(0, dtype=np.int32)
                data = np.empty(0, dtype=np.float64)
            return sp.csr_matrix((data, (rows, cols)), shape=(n, n))

        self._csr = _to_csr(best)
        if timed_all and best_t:
            self._csr_t = _to_csr({k: v[0] for k, v in best_t.items()})
            self._time_edge_len = {k: v[1] for k, v in best_t.items()}
        else:
            self._csr_t = None
            self._time_edge_len = None
        self._csr_row = row_of
        self._csr_nodes = np.array(node_ids)

    def register_route_targets(self, latlons: Sequence[LatLon]) -> None:
        """Register (lat, lon) points as routing targets.

        Single-source Dijkstra results are stored only for registered target
        nodes — the closed set of places routing ever ends at (agent homes and
        work places, catalog POIs). Intermediate intersections are just passed
        through and their distances are never queried, so at metro scale
        pruning cuts each cached source tree from ~all-nodes to ~|targets|.
        Call this once up front; unregistered destinations still self-heal
        lazily in :meth:`route_length_km` (one extra Dijkstra per stale source).
        """
        added = False
        for lat, lon in latlons:
            node = self.nearest_node(lat, lon)
            if node in self._route_targets:
                continue
            # Nodes the precomputed matrix already covers need no target entry —
            # and skipping them is what stops one run's agent population from
            # bumping _targets_version and invalidating every other run's cache.
            if self._matrix_index(node) >= 0:
                continue
            self._route_targets.add(node)
            added = True
        if added:
            self._targets_version += 1

    # ── precomputed matrix lookups ───────────────────────────────────────────

    def _matrix_index(self, node: int) -> int:
        """Row/column of ``node`` in the precomputed matrices, or -1."""
        nodes = self._mat_nodes
        if nodes is None:
            return -1
        i = int(np.searchsorted(nodes, node))
        if i < nodes.size and int(nodes[i]) == int(node):
            return i
        return -1

    def _matrix_lookup(self, table, orig: int, dest: int,
                       tag: str = "") -> Optional[float]:
        """One cell of a precomputed table, or None when it cannot serve the pair.

        None means "ask the lazy path" — either no matrix is attached or one of
        the endpoints is outside the universe. An unreachable pair is *not* None:
        it is a stored ``inf``, which the caller turns into the same haversine
        fallback the lazy path uses.

        The source row is materialised on first touch and reused, because that is
        how the model asks: one origin against many candidate destinations.
        """
        if table is None:
            return None
        i = self._matrix_index(orig)
        if i < 0:
            return None
        j = self._matrix_index(dest)
        if j < 0:
            return None

        key = (tag, i)
        row = self._mat_rows.get(key)
        if row is None:
            row = np.array(table[i, :])
            # Bound the cache by bytes, not entries, so a big metro does not use
            # proportionally more memory. Cleared wholesale rather than evicted
            # one at a time: a run's working set moves together, and this keeps
            # the hot path free of bookkeeping.
            if len(self._mat_rows) >= self._mat_row_cap():
                self._mat_rows.clear()
            self._mat_rows[key] = row
        return float(row[j])

    def _mat_row_cap(self) -> int:
        """Rows to keep cached — roughly 400 MB per table."""
        u = self._mat_nodes.size if self._mat_nodes is not None else 1
        return max(256, int(400_000_000 / max(u * 8, 1)))

    @property
    def has_routing_matrix(self) -> bool:
        return self._mat_dist_m is not None

    def _single_source(self, src: int) -> Dict[int, float]:
        """Shortest-path lengths (metres) from ``src`` to every *registered
        target* node, cached per source. Computed with scipy csgraph Dijkstra
        (C-level); the full distance array is pruned to targets before storing.
        A cached entry is recomputed if targets were registered after it was
        built (version check)."""
        entry = self._ss_cache.get(src)
        if entry is not None and entry[0] == self._targets_version:
            return entry[1]
        from scipy.sparse.csgraph import dijkstra

        self._ensure_csr()
        dist = dijkstra(self._csr, directed=True, indices=self._csr_row[src])
        lengths: Dict[int, float] = {src: 0.0}
        row_of = self._csr_row
        for target in self._route_targets:
            row = row_of.get(target)
            if row is None:
                continue
            d = dist[row]
            if np.isfinite(d):
                lengths[int(target)] = float(d)
        self._ss_cache[src] = (self._targets_version, lengths)
        return lengths

    def _ensure_time_edge_index(self) -> None:
        """Build (once) the sorted ``row_u * n + row_v -> metres`` index over the
        time-weighted graph's edges, so predecessor edges can be resolved for
        every node in a single vectorised lookup."""
        if self._tlen_keys is not None:
            return
        self._ensure_csr()
        if self._time_edge_len is None:
            self._tlen_keys = np.empty(0, dtype=np.int64)
            self._tlen_vals = np.empty(0, dtype=np.float64)
            return
        n = len(self._csr_nodes)
        items = self._time_edge_len
        keys = np.fromiter((u * n + v for (u, v) in items),
                           dtype=np.int64, count=len(items))
        vals = np.fromiter(items.values(), dtype=np.float64, count=len(items))
        order = np.argsort(keys, kind="stable")
        self._tlen_keys = keys[order]
        self._tlen_vals = vals[order]

    def fastest_path_meters(self, src_row: int, seconds, preds):
        """Metres along the fastest path from ``src_row`` to **every** node."""
        self._ensure_time_edge_index()
        return fastest_path_meters(self._tlen_keys, self._tlen_vals, seconds, preds)

    def _single_source_time(self, src: int) -> Optional[Dict[int, Tuple[float, float]]]:
        """Fastest-path ``(minutes, km)`` from ``src`` to every registered
        target, cached per source (None when edges carry no travel_time).

        One time-weighted Dijkstra with predecessors; km is the length of the
        *driven* (fastest) path — a freeway detour counts its real km — recovered
        for all nodes at once by :meth:`fastest_path_meters`."""
        entry = self._ss_time_cache.get(src)
        if entry is not None and entry[0] == self._targets_version:
            return entry[1]
        from scipy.sparse.csgraph import dijkstra

        self._ensure_csr()
        if self._csr_t is None:
            return None
        src_row = self._csr_row[src]
        seconds, preds = dijkstra(
            self._csr_t, directed=True, indices=src_row, return_predecessors=True
        )
        meters = self.fastest_path_meters(src_row, seconds, preds)
        out: Dict[int, Tuple[float, float]] = {src: (0.0, 0.0)}
        for target in self._route_targets:
            row = self._csr_row.get(target)
            if row is None or not np.isfinite(seconds[row]):
                continue
            out[int(target)] = (float(seconds[row]) / 60.0, float(meters[row]) / 1000.0)
        self._ss_time_cache[src] = (self._targets_version, out)
        return out

    def route_car_latlon(self, a: LatLon, b: LatLon) -> Optional[Tuple[float, float]]:
        """Fastest-path ``(free-flow minutes, km)`` between two points for car
        routing, or None when the graph has no travel_time data (callers fall
        back to distance / flat mode speed). Cached; targets self-heal like
        :meth:`route_length_km`. Congestion and weather scale the minutes
        downstream — this is the uncongested network time."""
        orig = self.nearest_node(*a)
        dest = self.nearest_node(*b)
        if orig == dest:
            return (0.0, 0.0)
        key = (orig, dest)
        cached = self._car_cache.get(key)
        if cached is None:
            seconds = self._matrix_lookup(self._mat_car_sec, orig, dest, "s")
            if seconds is not None:
                meters = self._matrix_lookup(self._mat_car_m, orig, dest, "m")
                if np.isfinite(seconds) and meters is not None and np.isfinite(meters):
                    cached = (seconds / 60.0, meters / 1000.0)
                else:  # unreachable (shouldn't happen on the SCC core)
                    cached = (None, haversine_km(self.node_latlon(orig),
                                                 self.node_latlon(dest)))
            else:
                if dest not in self._route_targets:
                    self._route_targets.add(dest)
                    self._targets_version += 1
                times = self._single_source_time(orig)
                if times is None:  # graph carries no travel_time at all
                    return None
                cached = times.get(dest)
                if cached is None:  # unreachable
                    km = haversine_km(self.node_latlon(orig), self.node_latlon(dest))
                    cached = (None, km)
            self._car_cache[key] = cached
        if cached[0] is None:
            return None
        return cached

    def _route_weight(self) -> str:
        """The metric the model routes by: fastest path when edge times exist
        (how car km/time are measured), shortest otherwise."""
        return "travel_time" if getattr(self, "_has_edge_times", False) else "length"

    def _route_nodes(self, orig: int, dest: int) -> Optional[List[int]]:
        """Node sequence of the routed path, or None when unreachable.

        Uses the same scipy CSR the distance queries use, rather than
        ``osmnx.routing.shortest_path`` (networkx), which was ~411 ms per route
        on a metro graph against ~15 ms for one scipy single-source pass. The
        predecessor array is cached per origin, so every further destination
        from the same origin costs only a pointer walk — which is exactly how
        the day-planner asks.

        A second benefit is consistency: the precomputed ``car_m`` table measures
        km along *this* predecessor tree's path, so the polyline now depicts the
        route whose distance the model actually charged. osmnx could previously
        return a different equal-cost path.
        """
        self._ensure_csr()
        csr = self._csr_t if self._csr_t is not None else self._csr
        src_row = self._csr_row.get(orig)
        dst_row = self._csr_row.get(dest)
        if src_row is None or dst_row is None:
            return None

        preds = self._pred_cache.get(src_row)
        if preds is None:
            from scipy.sparse.csgraph import dijkstra

            _dist, preds = dijkstra(csr, directed=True, indices=src_row,
                                    return_predecessors=True)
            # Bounded like the matrix row cache: an array per origin is n*4
            # bytes, so a few hundred is tens of MB.
            if len(self._pred_cache) >= 512:
                self._pred_cache.clear()
            self._pred_cache[src_row] = preds

        rows = [dst_row]
        r = dst_row
        while r != src_row:
            p = int(preds[r])
            if p < 0:
                return None  # unreachable
            rows.append(p)
            r = p
        rows.reverse()
        nodes = self._csr_nodes
        return [int(nodes[i]) for i in rows]

    def _edge_points(self, u: int, v: int, weight: str) -> List[LatLon]:
        """Coordinates along the (u, v) edge the router would take.

        Picks the minimum-weight parallel edge — the same one ``_ensure_csr``
        collapsed into the CSR — and orients its geometry u -> v, since OSM ways
        are not always digitised in the direction of travel.
        """
        data = min(self.G[u][v].values(), key=lambda d: float(d.get(weight, 1.0)))
        geom = data.get("geometry")
        if geom is None:
            return [self.node_latlon(u), self.node_latlon(v)]
        coords = [(round(y, 6), round(x, 6)) for x, y in geom.coords]
        if len(coords) > 1:
            u_ll = self.node_latlon(u)
            head = (coords[0][0] - u_ll[0]) ** 2 + (coords[0][1] - u_ll[1]) ** 2
            tail = (coords[-1][0] - u_ll[0]) ** 2 + (coords[-1][1] - u_ll[1]) ** 2
            if tail < head:
                coords.reverse()
        return coords

    def route_geometry_latlon(self, a: LatLon, b: LatLon) -> List[LatLon]:
        """Return the route polyline as ``[(lat, lon), ...]`` for visualization."""
        orig = self.nearest_node(*a)
        dest = self.nearest_node(*b)
        if orig == dest:
            return [self.node_latlon(orig)]
        cached = self._geom_cache.get((orig, dest))
        if cached is not None:
            return cached

        weight = self._route_weight()
        route = self._route_nodes(orig, dest)
        if not route or len(route) < 2:
            pts = [self.node_latlon(orig), self.node_latlon(dest)]
        else:
            pts = []
            for u, v in zip(route[:-1], route[1:]):
                try:
                    seg = self._edge_points(u, v, weight)
                except (KeyError, ValueError):
                    seg = [self.node_latlon(u), self.node_latlon(v)]
                for latlon in seg:
                    if not pts or pts[-1] != latlon:
                        pts.append(latlon)
        pts = pts or [self.node_latlon(orig), self.node_latlon(dest)]
        self._geom_cache[(orig, dest)] = pts
        return pts

    # ── POI insertion (mid-block nodes) ──────────────────────────────────────

    def _edge_line(self, u: int, v: int, k) -> LineString:
        """LineString for an edge — its geometry if present, else a straight segment."""
        data = self.G.edges[u, v, k]
        geom = data.get("geometry")
        if geom is not None:
            return geom
        return LineString(
            [(self.G.nodes[u]["x"], self.G.nodes[u]["y"]),
             (self.G.nodes[v]["x"], self.G.nodes[v]["y"])]
        )

    def _new_node_id(self) -> int:
        nid = self._next_poi_node_id
        self._next_poi_node_id += 1
        return nid

    def _refresh_nodes(self) -> None:
        """Rebuild node coordinate tables and clear routing caches after edits."""
        self.nodes = list(self.G.nodes)
        self._lat, self._lon = {}, {}
        for n, d in self.G.nodes(data=True):
            self._lat[n] = float(d["y"])
            self._lon[n] = float(d["x"])
        self._snap_cache.clear()
        self._dist_cache.clear()
        self._ss_cache.clear()
        self._ss_time_cache.clear()
        self._car_cache.clear()
        self._geom_cache.clear()
        self._reach_cache.clear()
        # Acceleration structures must be rebuilt against the new node set.
        self._kdtree = None
        self._kdtree_nodes = None
        self._csr = None
        self._csr_row = None
        self._csr_nodes = None
        self._csr_t = None
        self._time_edge_len = None
        self._tlen_keys = None
        self._tlen_vals = None
        self._pred_cache = {}
        # Node ids changed, so any precomputed matrix indexed by them is void.
        self._mat_nodes = None
        self._mat_dist_m = None
        self._mat_car_sec = None
        self._mat_car_m = None

    def _split_edge_through(self, a: int, b: int, k, poi_node_ids: List[int]) -> None:
        """Replace directed edge (a,b,k) with a chain a → … → b through the given
        POI nodes (ordered by their projection along the edge), preserving total
        length and attributes."""
        G = self.G
        data = G.edges[a, b, k]
        line = self._edge_line(a, b, k)
        total = line.length or 1e-12
        orig_len = data.get("length")
        if orig_len is None:
            orig_len = haversine_km(
                (G.nodes[a]["y"], G.nodes[a]["x"]), (G.nodes[b]["y"], G.nodes[b]["x"])
            ) * 1000.0
        # travel_time is length-proportional, so each segment gets its share
        # (copying the whole edge's time to every segment would overcount).
        orig_time = data.get("travel_time")
        attrs = {kk: vv for kk, vv in data.items()
                 if kk not in ("geometry", "length", "travel_time")}

        # Rank breaks ties: the origin endpoint must sort first and the far
        # endpoint last even when a POI projects exactly onto one of them —
        # otherwise that POI node ends up with no outgoing (or incoming) edge
        # on a one-way street and becomes unreachable.
        positions = [(0.0, 0, a), (total, 2, b)]
        for nid in poi_node_ids:
            d = line.project(Point(G.nodes[nid]["x"], G.nodes[nid]["y"]))
            positions.append((min(max(d, 0.0), total), 1, nid))
        positions.sort(key=lambda e: (e[0], e[1]))

        for (da, _ra, an), (db, _rb, bn) in zip(positions[:-1], positions[1:]):
            if an == bn:
                continue
            seg = dict(attrs)
            seg["length"] = max(0.1, orig_len * (db - da) / total)
            if orig_time is not None:
                seg["travel_time"] = max(0.01, float(orig_time) * (db - da) / total)
            try:
                sub = substring(line, da, db)
                if sub.length > 0:
                    seg["geometry"] = sub
            except Exception:
                pass
            G.add_edge(an, bn, **seg)
        G.remove_edge(a, b, k)

    def add_pois_as_nodes(self, points) -> Dict[str, Tuple[int, float, float]]:
        """Insert POIs into the graph as mid-block nodes by splitting their nearest
        edge, so routing to/from a POI is granular within a block (rather than
        snapping to the nearest intersection).

        ``points``: iterable of ``(poi_id, lat, lon)``. Idempotent per id. Returns
        ``{poi_id: (node_id, lat, lon)}`` with the on-street snapped coordinate.
        Both directions of a two-way street are split, and multiple POIs on the
        same block chain along it.
        """
        pts = [(str(pid), float(lat), float(lon)) for pid, lat, lon in points]
        todo = [(pid, lat, lon) for pid, lat, lon in pts if pid not in self._poi_nodes]

        if todo:
            G = self.G
            edge_keys = list(G.edges(keys=True))
            tree = STRtree([self._edge_line(*ek) for ek in edge_keys])

            # Group POIs by undirected street so we split each street exactly once
            # (a two-way street is two directed edges sharing one physical road).
            per_street: Dict[frozenset, dict] = {}
            for pid, lat, lon in todo:
                hit = tree.query_nearest(Point(lon, lat), all_matches=False)
                idx = int(hit[0]) if hasattr(hit, "__len__") else int(hit)
                u, v, k = edge_keys[idx]
                street = per_street.setdefault(frozenset((u, v)), {"rep": (u, v, k), "pts": []})
                street["pts"].append((pid, lat, lon))

            for info in per_street.values():
                u, v, k = info["rep"]
                if not G.has_edge(u, v, k):
                    continue
                line = self._edge_line(u, v, k)
                poi_nids = []
                for pid, lat, lon in info["pts"]:
                    pt = line.interpolate(line.project(Point(lon, lat)))
                    nid = self._new_node_id()
                    G.add_node(nid, x=float(pt.x), y=float(pt.y))
                    self._poi_nodes[pid] = nid
                    poi_nids.append(nid)
                self._split_edge_through(u, v, k, poi_nids)
                if G.has_edge(v, u):
                    for rk in list(G.get_edge_data(v, u).keys()):
                        self._split_edge_through(v, u, rk, poi_nids)

            self._refresh_nodes()

        return {
            pid: (self._poi_nodes[pid], *self.node_latlon(self._poi_nodes[pid]))
            for pid, _, _ in pts if pid in self._poi_nodes
        }

    # ── two-layer metro classification (attach_layers) ───────────────────────

    def attach_layers(self, core_polygon, county_polygons: Dict[str, object],
                      county_meta: Dict[str, dict]) -> None:
        """Classify base nodes into the two abstract layers of a metro network.

        One graph, two layers: nodes inside ``core_polygon`` (the principal
        city) form the pool for work locations and POI placement; every county
        polygon gets a core/shell node-pool split for home sampling.

        ``county_meta``: ``{fips: {"name", "workers", "core_share"}}`` —
        ``workers`` weights county choice when sampling homes; ``core_share``
        is the probability a home in that county falls inside the core city
        (provisional until ACS block-level homes arrive; see datasource.py).
        """
        import shapely

        lats = np.fromiter((self._lat[n] for n in self._base_nodes),
                           dtype=np.float64, count=len(self._base_nodes))
        lons = np.fromiter((self._lon[n] for n in self._base_nodes),
                           dtype=np.float64, count=len(self._base_nodes))
        ids = np.array(self._base_nodes)

        shapely.prepare(core_polygon)
        core_mask = shapely.contains_xy(core_polygon, lons, lats)
        self.core_polygon = core_polygon
        self._core_nodes = [int(n) for n in ids[core_mask]]
        if not self._core_nodes:
            raise ValueError(
                f"core polygon of {self.name!r} contains no network nodes"
            )

        unassigned = np.ones(len(ids), dtype=bool)
        pools: Dict[str, Dict[str, List[int]]] = {}
        for fips, poly in county_polygons.items():
            shapely.prepare(poly)
            mask = shapely.contains_xy(poly, lons, lats) & unassigned
            unassigned &= ~mask
            pools[fips] = {
                "core": [int(n) for n in ids[mask & core_mask]],
                "shell": [int(n) for n in ids[mask & ~core_mask]],
            }
        self._county_pools = pools
        self.county_meta = {f: dict(m) for f, m in county_meta.items()}
        self._county_ids = [
            f for f in pools
            if (pools[f]["core"] or pools[f]["shell"])
            and self.county_meta.get(f, {}).get("workers", 0) > 0
        ]
        self._county_weights = [
            float(self.county_meta[f]["workers"]) for f in self._county_ids
        ]

        core_lats, core_lons = lats[core_mask], lons[core_mask]
        self._core_bounds = (float(core_lats.min()), float(core_lons.min()),
                             float(core_lats.max()), float(core_lons.max()))
        self._core_center = (float(core_lats.mean()), float(core_lons.mean()))

    @property
    def has_layers(self) -> bool:
        return self._core_nodes is not None

    @property
    def core_bounds(self) -> BBox:
        """(south, west, north, east) of the core; full bounds when unlayered."""
        return self._core_bounds if self._core_bounds is not None else self.bounds

    @property
    def core_center(self) -> LatLon:
        return self._core_center if self._core_center is not None else self.center

    def in_core(self, lat: float, lon: float) -> bool:
        """True when a point lies in the principal city (always True unlayered)."""
        if self.core_polygon is None:
            return True
        import shapely

        return bool(shapely.contains_xy(self.core_polygon, lon, lat))

    def in_core_mask(self, lats: Sequence[float], lons: Sequence[float]):
        """Vectorised :meth:`in_core` for bulk POI clipping."""
        if self.core_polygon is None:
            return np.ones(len(lats), dtype=bool)
        import shapely

        shapely.prepare(self.core_polygon)
        return shapely.contains_xy(
            self.core_polygon, np.asarray(lons, dtype=np.float64),
            np.asarray(lats, dtype=np.float64),
        )

    # ── sampling ───────────────────────────────────────────────────────────--

    def sample_node_latlon(self, rng: random.Random) -> LatLon:
        """Return a uniformly random *intersection* node coordinate.

        Samples only original road nodes (not inserted POI nodes), so
        homes/work/organic destinations stay on real intersections.
        """
        return self.node_latlon(rng.choice(self._base_nodes))

    def sample_core_latlon(self, rng: random.Random) -> LatLon:
        """Random intersection inside the principal city (work locations,
        synthetic POIs, fallback destinations). Any intersection unlayered."""
        if not self.has_layers:
            return self.sample_node_latlon(rng)
        return self.node_latlon(rng.choice(self._core_nodes))

    def sample_home_latlon(self, rng: random.Random) -> LatLon:
        """Random home: county drawn ∝ inbound workers, then core vs rest of
        county by that county's ``core_share``. Any intersection unlayered.

        FALLBACK ONLY. This is the coarse placement used when no real commute
        data is attached — the county weights say *where commuters live*, but
        within a pool nodes are uniform. :meth:`sample_commute` supersedes it
        whenever ``metro_od_pairs`` is available.
        """
        if not self.has_layers:
            return self.sample_node_latlon(rng)
        fips = rng.choices(self._county_ids, weights=self._county_weights, k=1)[0]
        pool = self._county_pools[fips]
        share = float(self.county_meta[fips].get("core_share", 0.0))
        nodes = pool["core"] if (pool["core"] and rng.random() < share) else pool["shell"]
        if not nodes:
            nodes = pool["core"] or pool["shell"]
        return self.node_latlon(rng.choice(nodes))

    # ── real commutes (LODES block pairs) ────────────────────────────────────

    @staticmethod
    def lodes_age_segment(age_years) -> int:
        """Index of the LODES worker-age segment holding ``age_years``.

        LODES OD tables are segmented as sa01 (<=29), sa02 (30-54), sa03 (55+);
        the boundaries here are those, not the survey's own age bands.
        """
        if age_years <= 29:
            return 0
        if age_years <= 54:
            return 1
        return 2

    @staticmethod
    def lodes_earnings_segment(annual_income) -> int:
        """Index of the LODES earnings segment for an annual income figure.

        LODES segments monthly earnings as se01 (<=$1250), se02 ($1251-3333),
        se03 (>$3333), so the annual figure is divided by 12.

        Caveat worth carrying: LODES earnings are what one worker is paid for
        one job, while the survey reports total HOUSEHOLD income before taxes.
        Household income is at least individual earnings and usually more, so
        agents from multi-earner households land in a higher segment than their
        own pay would place them, biasing the conditioned draw toward the
        workplaces of higher earners. Callers pass this only for agents who hold
        a job; a non-earner has no job earnings to segment on at all.
        """
        monthly = float(annual_income) / 12.0
        if monthly <= 1250.0:
            return 0
        if monthly <= 3333.0:
            return 1
        return 2

    def _commute_segment_cum(self, age_seg, earn_seg):
        """Cumulative job weights within a LODES age and/or earnings segment.

        Built on first use and cached under ``(age_seg, earn_seg)``: a metro
        carries millions of pairs and a run touches at most a few thousand, so
        the prefix sums are only worth paying for in the segments a population
        actually draws from. Either index may be None to leave that margin free.

        LODES publishes the two segmentations as separate marginal counts per
        pair, never their cross-tabulation, so conditioning on both at once
        cannot be exact. The joint weight is taken as
        ``age[a] * earn[e] / jobs``, the count expected if the two were
        independent within the pair. Conditioning on a single margin needs no
        such assumption.

        Returns ``(None, 0.0)`` when the requested segment carries no jobs, which
        is the caller's signal to fall back to the unconditioned draw.
        """
        key = (age_seg, earn_seg)
        cached = self._commute_seg_cum.get(key)
        if cached is not None:
            return cached

        pairs = self._commutes
        age = getattr(pairs, "age", None)
        earn = getattr(pairs, "earn", None)
        n = len(pairs)

        def margin(arr, index):
            if index is None:
                return None
            if arr is None or arr.shape[0] != n:
                return False  # segment counts unavailable
            return np.asarray(arr[:, index], dtype=np.float64)

        a = margin(age, age_seg)
        e = margin(earn, earn_seg)
        if a is False or e is False:
            entry = (None, 0.0)
        else:
            if a is None and e is None:
                weights = np.asarray(pairs.jobs, dtype=np.float64)
            elif e is None:
                weights = a
            elif a is None:
                weights = e
            else:
                jobs = np.asarray(pairs.jobs, dtype=np.float64)
                weights = np.divide(a * e, jobs, out=np.zeros_like(jobs), where=jobs > 0)
            cum = np.cumsum(weights)
            total = float(cum[-1]) if cum.size else 0.0
            entry = (cum, total) if total > 0 else (None, 0.0)

        self._commute_seg_cum[key] = entry
        return entry

    def attach_commutes(self, pairs) -> None:
        """Attach block-level home->work pairs as the placement distribution.

        ``pairs`` is a :class:`welfare_rs.datasource.CommutePairs` (or None to
        clear). Nothing is snapped here: a metro can hold millions of pairs and
        a run draws at most a few thousand, so snapping is deferred to
        :meth:`sample_commute` and served by the existing BallTree + snap cache.
        """
        if pairs is None or len(pairs) == 0:
            self._commutes = None
            self._commute_cum = None
            self._commute_total = 0.0
            self._commute_seg_cum = {}
            return
        weights = np.asarray(pairs.jobs, dtype=np.float64)
        cum = np.cumsum(weights)
        total = float(cum[-1])
        if total <= 0:
            self._commutes = None
            self._commute_cum = None
            self._commute_total = 0.0
            self._commute_seg_cum = {}
            return
        self._commutes = pairs
        self._commute_cum = cum
        self._commute_total = total
        self._commute_seg_cum = {}

    def attach_home_blocks(self, points) -> None:
        """Record populated core block coordinates as possible agent homes.

        ``points`` is a sequence of ``(lat, lon)``, or None to clear. Nothing
        is snapped here — the routing precompute bulk-snaps them once, which is
        far cheaper than snapping each on attach.
        """
        self._home_block_latlon = list(points) if points else None

    @property
    def home_block_latlon(self):
        """Core block coordinates a non-worker agent home can be drawn at."""
        return self._home_block_latlon or []

    @property
    def has_commutes(self) -> bool:
        return self._commutes is not None

    @property
    def num_commute_pairs(self) -> int:
        return 0 if self._commutes is None else len(self._commutes)

    def sample_commute(self, rng: random.Random, age_years=None, annual_income=None,
                       with_true: bool = False):
        """Draw one real commute, or None when no pairs are attached.

        A pair is drawn with probability ∝ its LODES job count, so the commute
        length distribution — and the home/work correlation that produces it —
        matches the published data rather than two independent uniform draws.

        Passing ``age_years`` and/or ``annual_income`` restricts the draw to the
        matching LODES worker segment, so an agent's commute geography stays
        consistent with the demographics it was given: where the young and the
        over-55 work in a metro, and where low and high earners work, are not the
        same distributions. Falls back to the unconditioned draw when the segment
        carries no jobs, or when the attached pairs have no segment counts.

        ``annual_income`` is divided by 12 to reach the monthly figure LODES
        segments on. See :meth:`lodes_earnings_segment` for why that mapping is
        approximate, and :meth:`_commute_segment_cum` for how the two margins are
        combined when both are given.

        Returns ``(home, work, home_access_km, work_access_km)``. The two
        locations are the block internal points **snapped to the graph**; the
        access distances are how far each snap moved, straight-line. That gap
        is not noise to be discarded: outside the core the graph is arterials
        only, so a suburban home can snap 1–3 km to the nearest arterial. The
        caller charges that distance as an un-networked access leg
        (:meth:`welfare_rs.Simulation._leg_outcomes`) instead of letting the
        commute silently start from the arterial.

        ``with_true`` appends the two unsnapped block points, which the walk and
        bike networks snap on their own.
        """
        if self._commutes is None:
            return None
        cum, total = self._commute_cum, self._commute_total
        if age_years is not None or annual_income is not None:
            seg_cum, seg_total = self._commute_segment_cum(
                None if age_years is None else self.lodes_age_segment(age_years),
                None if annual_income is None else self.lodes_earnings_segment(annual_income),
            )
            if seg_total > 0:
                cum, total = seg_cum, seg_total
        x = rng.random() * total
        i = int(np.searchsorted(cum, x, side="right"))
        i = min(i, len(cum) - 1)
        c = self._commutes

        h_true = (float(c.h_lat[i]), float(c.h_lon[i]))
        w_true = (float(c.w_lat[i]), float(c.w_lon[i]))
        home = self.snap_latlon(*h_true)
        work = self.snap_latlon(*w_true)
        drawn = (home, work, haversine_km(h_true, home), haversine_km(w_true, work))
        return drawn + (h_true, w_true) if with_true else drawn

    # ── stats ────────────────────────────────────────────────────────────────

    @property
    def num_nodes(self) -> int:
        return self.G.number_of_nodes()

    @property
    def num_base_nodes(self) -> int:
        """Count of original road (intersection) nodes, excluding inserted POIs."""
        return len(self._base_nodes)

    @property
    def num_edges(self) -> int:
        return self.G.number_of_edges()

    def __repr__(self) -> str:
        return (
            f"RoadNetwork(name={self.name!r}, type={self.network_type!r}, "
            f"nodes={self.num_nodes}, edges={self.num_edges})"
        )


def build_road_network(
    city: Optional[str] = None,
    *,
    cache_dir: Optional[str] = None,
    network_type: Optional[str] = None,
    active: bool = True,
) -> RoadNetwork:
    """Build (or load from cache) a :class:`RoadNetwork` from a params preset.

    Shared by the frontend, the persona generator, and any script so they all
    agree on the same area and cache. See ``params.GEO_PARAMS``. With
    ``active`` (the default) the walk and bike networks for the same box are
    attached as ``mode_networks``; tooling that never simulates can skip them.
    """
    from . import params

    gp = params.GEO_PARAMS
    city = city or gp["default_city"]
    spec = gp["cities"][city]
    net = RoadNetwork(
        city,
        bbox=spec.get("bbox"),
        center=spec.get("center"),
        dist_m=spec.get("dist_m"),
        network_type=network_type or gp["default_network_type"],
        cache_dir=cache_dir or gp["cache_dir"],
    )
    # Walking and cycling get their own networks over the same box, so a demo
    # on a preset runs the same mode choice a metro does.
    if active and spec.get("bbox") is not None and net.network_type == "drive":
        from .active_modes import attach_bbox_active_networks

        attach_bbox_active_networks(net, spec["bbox"], cache_dir=net.cache_dir)
    return net

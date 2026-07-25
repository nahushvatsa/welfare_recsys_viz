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
* All modes (car, walk, bike, transit) route on this single drive network; the
  mode differences (speed, wait, cost, comfort) live in ``params.MODE_PARAMS``
  and are applied downstream. This is the generalisation of "walking can be the
  same route as the car, with scaled time".
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
        if self._ensure_travel_times() and graph is None:
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

    def __getstate__(self) -> dict:
        """Exclude the lazily-rebuilt acceleration structures from pickling (so
        the disk cache stays small and isn't coupled to sklearn/scipy pickle
        versions); they rebuild on first use after a load. The graph + routing
        caches (the expensive, reusable parts) are persisted."""
        state = self.__dict__.copy()
        for k in ("_kdtree", "_kdtree_nodes", "_csr", "_csr_row", "_csr_nodes",
                  "_csr_t", "_time_edge_len"):
            state[k] = None
        return state

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
        if dist is None:
            if dest not in self._route_targets:
                self._route_targets.add(dest)
                self._targets_version += 1
            meters = self._single_source(orig).get(dest)
            if meters is None:  # unreachable (shouldn't happen on the routable core)
                dist = haversine_km(self.node_latlon(orig), self.node_latlon(dest))
            else:
                dist = meters / 1000.0
            self._dist_cache[key] = dist
        return dist

    def route_length_km_latlon(self, a: LatLon, b: LatLon) -> float:
        return self.route_length_km(self.nearest_node(*a), self.nearest_node(*b))

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
            if node not in self._route_targets:
                self._route_targets.add(node)
                added = True
        if added:
            self._targets_version += 1

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

    def _single_source_time(self, src: int) -> Optional[Dict[int, Tuple[float, float]]]:
        """Fastest-path ``(minutes, km)`` from ``src`` to every registered
        target, cached per source (None when edges carry no travel_time).

        One time-weighted Dijkstra with predecessors; km is accumulated by
        walking each target's predecessor chain, so it is the length of the
        *driven* (fastest) path — a freeway detour counts its real km."""
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
        out: Dict[int, Tuple[float, float]] = {src: (0.0, 0.0)}
        edge_len = self._time_edge_len
        for target in self._route_targets:
            row = self._csr_row.get(target)
            if row is None or not np.isfinite(seconds[row]):
                continue
            meters, r = 0.0, row
            while r != src_row:
                p = int(preds[r])
                if p < 0:
                    meters = -1.0
                    break
                meters += edge_len[(p, r)]
                r = p
            if meters >= 0.0:
                out[int(target)] = (float(seconds[row]) / 60.0, meters / 1000.0)
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
            if dest not in self._route_targets:
                self._route_targets.add(dest)
                self._targets_version += 1
            times = self._single_source_time(orig)
            if times is None:
                return None
            cached = times.get(dest)
            if cached is None:  # unreachable (shouldn't happen on the SCC core)
                km = haversine_km(self.node_latlon(orig), self.node_latlon(dest))
                cached = (None, km)
            self._car_cache[key] = cached
        if cached[0] is None:
            return None
        return cached

    def route_geometry_latlon(self, a: LatLon, b: LatLon) -> List[LatLon]:
        """Return the route polyline as ``[(lat, lon), ...]`` for visualization."""
        orig = self.nearest_node(*a)
        dest = self.nearest_node(*b)
        if orig == dest:
            return [self.node_latlon(orig)]
        cached = self._geom_cache.get((orig, dest))
        if cached is not None:
            return cached
        # Match the metric the model routes by: fastest path when edge times
        # exist (how car km/time are measured), shortest otherwise.
        weight = "travel_time" if getattr(self, "_has_edge_times", False) else "length"
        route = ox.routing.shortest_path(self.G, orig, dest, weight=weight)
        if not route or len(route) < 2:
            pts = [self.node_latlon(orig), self.node_latlon(dest)]
        else:
            try:
                gdf = ox.routing.route_to_gdf(self.G, route, weight=weight)
                pts = []
                for geom in gdf["geometry"]:
                    for x, y in geom.coords:  # shapely stores coordinates as (lon, lat)
                        latlon = (round(y, 6), round(x, 6))
                        if not pts or pts[-1] != latlon:
                            pts.append(latlon)
            except Exception:
                pts = [self.node_latlon(n) for n in route]
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
        # Acceleration structures must be rebuilt against the new node set.
        self._kdtree = None
        self._kdtree_nodes = None
        self._csr = None
        self._csr_row = None
        self._csr_nodes = None
        self._csr_t = None
        self._time_edge_len = None

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

        Provisional placement — the county weights say *where commuters live*;
        within a pool nodes are uniform until ACS block-level homes replace
        this via the datasource.
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
) -> RoadNetwork:
    """Build (or load from cache) a :class:`RoadNetwork` from a params preset.

    Shared by the frontend, the persona generator, and any script so they all
    agree on the same area and cache. See ``params.GEO_PARAMS``.
    """
    from . import params

    gp = params.GEO_PARAMS
    city = city or gp["default_city"]
    spec = gp["cities"][city]
    return RoadNetwork(
        city,
        bbox=spec.get("bbox"),
        center=spec.get("center"),
        dist_m=spec.get("dist_m"),
        network_type=network_type or gp["default_network_type"],
        cache_dir=cache_dir or gp["cache_dir"],
    )

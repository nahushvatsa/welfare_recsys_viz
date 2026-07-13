"""OSM road-network backend for the travel-behaviour ABM.

This module replaces the synthetic grid in :mod:`city` with a real street
network downloaded from OpenStreetMap via OSMnx. It is deliberately free of any
ABM knowledge: it only snaps coordinates to network nodes, measures network
routes (length + geometry), and answers single-source distance queries used by
the agent day-planner.

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

import math
import os
import random
from typing import Dict, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
import osmnx as ox
from shapely import STRtree
from shapely.geometry import LineString, Point
from shapely.ops import substring

LatLon = Tuple[float, float]
BBox = Tuple[float, float, float, float]  # (west, south, east, north)

DEFAULT_CACHE_DIR = os.environ.get(
    "WELFARE_RS_CACHE",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache"),
)

_EARTH_RADIUS_KM = 6371.0088


def haversine_km(a: LatLon, b: LatLon) -> float:
    """Great-circle ("as the crow flies") distance in km between two points.

    Used by the recommender systems for their proximity heuristic, which the
    paper models as straight-line distance — deliberately cruder than the
    network routing used for the realised trip cost.
    """
    lat1, lon1 = a
    lat2, lon2 = b
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    h = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlmb / 2) ** 2
    return 2 * _EARTH_RADIUS_KM * math.asin(math.sqrt(h))


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

        self.G = self._load_or_download(
            bbox=bbox,
            center=center,
            dist_m=dist_m,
            simplify=simplify,
            strongly_connected=strongly_connected,
        )

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

        # Caches: coord->node snap, (orig,dest)->km, source->{node: meters},
        # (orig,dest)->route polyline.
        self._snap_cache: Dict[Tuple[float, float], int] = {}
        self._dist_cache: Dict[Tuple[int, int], float] = {}
        self._ss_cache: Dict[int, Dict[int, float]] = {}
        self._geom_cache: Dict[Tuple[int, int], List[LatLon]] = {}

        # Lazily-built acceleration structures (invalidated by _refresh_nodes):
        #   _kdtree: haversine BallTree over node coords — O(log n) snapping,
        #            built once instead of osmnx rebuilding its index per call.
        #   _csr:    CSR adjacency (min length per directed edge) — lets scipy
        #            run Dijkstra in C. Both yield identical results to before.
        self._kdtree = None
        self._kdtree_nodes = None  # np.ndarray: tree row -> node id
        self._csr = None
        self._csr_row: Optional[Dict[int, int]] = None   # node id -> CSR row
        self._csr_nodes = None     # np.ndarray: CSR row -> node id

        # Original road (intersection) nodes — homes/work/organic locations sample
        # only these. POIs may be inserted as extra mid-block nodes (see
        # add_pois_as_nodes) without polluting that pool.
        self._base_nodes: List[int] = list(self.nodes)
        self._poi_nodes: Dict[str, int] = {}
        self._next_poi_node_id = 10_000_000_000_000

    def __getstate__(self) -> dict:
        """Exclude the lazily-rebuilt acceleration structures from pickling (so
        the disk cache stays small and isn't coupled to sklearn/scipy pickle
        versions); they rebuild on first use after a load. The graph + routing
        caches (the expensive, reusable parts) are persisted."""
        state = self.__dict__.copy()
        for k in ("_kdtree", "_kdtree_nodes", "_csr", "_csr_row", "_csr_nodes"):
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
        """
        if orig == dest:
            return 0.0
        key = (orig, dest)
        dist = self._dist_cache.get(key)
        if dist is None:
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
        """Build (once) a CSR adjacency: the minimum ``length`` among parallel
        directed edges for each (u, v). networkx Dijkstra also uses the min
        parallel edge, so distances are identical — but scipy runs in C."""
        if self._csr is not None:
            return
        import scipy.sparse as sp

        node_ids = list(self.G.nodes)
        row_of = {n: i for i, n in enumerate(node_ids)}
        best: Dict[Tuple[int, int], float] = {}
        for u, v, d in self.G.edges(data=True):
            w = float(d.get("length", 1.0))
            key = (row_of[u], row_of[v])
            if key not in best or w < best[key]:
                best[key] = w
        n = len(node_ids)
        if best:
            keys = list(best.keys())
            rows = np.fromiter((k[0] for k in keys), dtype=np.int32, count=len(keys))
            cols = np.fromiter((k[1] for k in keys), dtype=np.int32, count=len(keys))
            data = np.fromiter((best[k] for k in keys), dtype=np.float64, count=len(keys))
        else:
            rows = cols = np.empty(0, dtype=np.int32)
            data = np.empty(0, dtype=np.float64)
        self._csr = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
        self._csr_row = row_of
        self._csr_nodes = np.array(node_ids)

    def _single_source(self, src: int) -> Dict[int, float]:
        """Shortest-path lengths (metres) from ``src`` to every reachable node,
        cached per source. Computed with scipy csgraph Dijkstra (C-level) instead
        of networkx — identical distances, ~10-20x faster."""
        lengths = self._ss_cache.get(src)
        if lengths is None:
            from scipy.sparse.csgraph import dijkstra

            self._ensure_csr()
            dist = dijkstra(self._csr, directed=True, indices=self._csr_row[src])
            nodes = self._csr_nodes
            finite = np.nonzero(np.isfinite(dist))[0]
            lengths = {int(nodes[i]): float(dist[i]) for i in finite}
            self._ss_cache[src] = lengths
        return lengths

    def distances_from_latlon(self, origin: LatLon, dests: Sequence[LatLon]) -> List[float]:
        """Network distances (km) from one origin to many destinations.

        One Dijkstra serves all destinations and is cached per source node — the
        agent day-planner scores many candidate leisure locations from the same
        (repeating) post-work origin, so this amortises well across days.
        """
        src = self.nearest_node(*origin)
        lengths = self._single_source(src)
        out: List[float] = []
        for dest in dests:
            node = self.nearest_node(*dest)
            if node in lengths:
                out.append(float(lengths[node]) / 1000.0)
            else:
                out.append(haversine_km(self.node_latlon(src), self.node_latlon(node)))
        return out

    def route_geometry_latlon(self, a: LatLon, b: LatLon) -> List[LatLon]:
        """Return the route polyline as ``[(lat, lon), ...]`` for visualization."""
        orig = self.nearest_node(*a)
        dest = self.nearest_node(*b)
        if orig == dest:
            return [self.node_latlon(orig)]
        cached = self._geom_cache.get((orig, dest))
        if cached is not None:
            return cached
        route = ox.routing.shortest_path(self.G, orig, dest, weight="length")
        if not route or len(route) < 2:
            pts = [self.node_latlon(orig), self.node_latlon(dest)]
        else:
            try:
                gdf = ox.routing.route_to_gdf(self.G, route, weight="length")
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
        self._geom_cache.clear()
        # Acceleration structures must be rebuilt against the new node set.
        self._kdtree = None
        self._kdtree_nodes = None
        self._csr = None
        self._csr_row = None
        self._csr_nodes = None

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
        attrs = {kk: vv for kk, vv in data.items() if kk not in ("geometry", "length")}

        positions = [(0.0, a), (total, b)]
        for nid in poi_node_ids:
            d = line.project(Point(G.nodes[nid]["x"], G.nodes[nid]["y"]))
            positions.append((min(max(d, 0.0), total), nid))
        positions.sort(key=lambda e: e[0])

        for (da, an), (db, bn) in zip(positions[:-1], positions[1:]):
            if an == bn:
                continue
            seg = dict(attrs)
            seg["length"] = max(0.1, orig_len * (db - da) / total)
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

    # ── sampling ───────────────────────────────────────────────────────────--

    def sample_node_latlon(self, rng: random.Random) -> LatLon:
        """Return a uniformly random *intersection* node coordinate.

        Samples only original road nodes (not inserted POI nodes), so
        homes/work/organic destinations stay on real intersections.
        """
        return self.node_latlon(rng.choice(self._base_nodes))

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

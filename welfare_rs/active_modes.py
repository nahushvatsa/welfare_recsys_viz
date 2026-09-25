"""Walk and bike networks for a metro's core.

The drive graph (:mod:`welfare_rs.metro`) cannot stand in for walking or
cycling: it has no footways, park paths, pedestrian streets or cycle tracks,
and it applies car one-way rules that do not bind a pedestrian. So each metro
gets two more OSM networks, restricted to the core because only core residents
walk or cycle in this model (anyone living outside it drives).

How each network is built
-------------------------
* **Extent.** The core polygon with its holes filled, plus ``buffer_m``. Filling
  holes matters for cores with enclaves — Beverly Hills and Culver City inside
  Los Angeles, Piedmont inside Oakland — because the real route between two
  parts of the city runs straight through them. The buffer lets a route use
  the streets just past the city line, and lets bridges reach the far bank:
  walking from Roosevelt Island to the rest of Manhattan really does go through
  Queens.
* **Walk** uses OSMnx's ``walk`` network: every edge in both directions. It
  keeps streets whose sidewalks are mapped as separate ways, which OSMnx would
  drop: those separate sidewalks are often not drawn across bridges and
  causeways, and dropping the street strands whatever lies beyond (Watson
  Island in Miami). The centreline runs beside its sidewalk, so distances do
  not change. It also adds cycleways where walking is explicitly allowed
  (``foot=designated`` shared-use trails), which OSMnx drops wholesale.
* **Bike** uses OSMnx's ``bike`` network and adds two things it misses. It
  includes footways where cycling is explicitly permitted (shared-use paths are
  often tagged ``highway=footway`` + ``bicycle=designated``), and it adds the
  reverse direction of one-way streets that allow contraflow cycling
  (``oneway:bicycle=no``). Without the second, cyclists would be routed round
  every contraflow lane.
* **Components.** The drive graph keeps only its largest strongly connected
  component. That would be wrong here: San Francisco and Oakland have no
  walkable or bikeable link across the Bay, so it would delete Oakland. Instead
  every sizeable piece is kept (its own largest strongly connected part), and
  only fragments under ``min_component_nodes`` are dropped. Those are the parking
  lot aisles and isolated park paths that would otherwise capture a home's snap
  and make every route from it unreachable. A pair in different pieces has no
  route, and routing reports that as "mode not available", never as a
  straight-line stand-in.

POIs are inserted as mid-block nodes, exactly as on the drive graph and from
the same selection, so a venue is reached at its street frontage.

Caches: ``<cache>/networks/<metro>_<mode>_<spec>.graphml`` (the downloaded,
cleaned graph) and ``<cache>/warmed/<metro>_<mode>.pkl`` (with POIs baked in),
plus routing matrices in ``<cache>/warmed/<metro>_<mode>_routing/``.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
from typing import Dict, List, Optional, Tuple

import networkx as nx
import osmnx as ox

from . import params, routing_matrix
from .geo import RoadNetwork

_PICKLE_PROTOCOL = 4

# Bumped when an active warm pickle changes shape.
ACTIVE_WARM_VERSION = 1


def _spec(mode: str) -> dict:
    """Every setting that changes the downloaded graph, for cache identity."""
    ap = params.ACTIVE_NETWORK_PARAMS
    spec = {
        "mode": mode,
        "buffer_m": float(ap["buffer_m"]),
        "min_component_nodes": int(ap["min_component_nodes"]),
    }
    if mode == "bike":
        spec["extra_filters"] = list(ap["bike_extra_filters"])
        spec["contraflow"] = bool(ap["bike_contraflow"])
    if mode == "walk":
        spec["keep_separate_sidewalk_streets"] = bool(ap["walk_keep_separate_sidewalk_streets"])
        spec["extra_filters"] = list(ap["walk_extra_filters"])
    return spec


def walk_filter() -> str:
    """OSMnx's walk filter, optionally without its separate-sidewalk clauses."""
    import re

    from osmnx import _overpass

    f = _overpass._get_network_filter("walk")
    if params.ACTIVE_NETWORK_PARAMS["walk_keep_separate_sidewalk_streets"]:
        f = re.sub(r'\["sidewalk(:[a-z]+)?"!~"separate"\]', "", f)
    return f


def _spec_hash(mode: str) -> str:
    raw = json.dumps(_spec(mode), sort_keys=True).encode("utf-8")
    return hashlib.md5(raw).hexdigest()[:10]


def _graphml_path(cache_dir: str, metro: str, mode: str) -> str:
    return os.path.join(cache_dir, "networks", f"{metro}_{mode}_{_spec_hash(mode)}.graphml")


def _warm_path(cache_dir: str, metro: str, mode: str) -> str:
    return os.path.join(cache_dir, "warmed", f"{metro}_{mode}.pkl")


def download_polygon(core_polygon):
    """Core polygon with holes filled, buffered by ``buffer_m``."""
    from shapely.geometry import MultiPolygon, Polygon
    from shapely.ops import unary_union

    parts = list(core_polygon.geoms) if isinstance(core_polygon, MultiPolygon) else [core_polygon]
    filled = unary_union([Polygon(p.exterior) for p in parts])
    deg = float(params.ACTIVE_NETWORK_PARAMS["buffer_m"]) / 111_000.0
    return filled.buffer(deg)


def _add_contraflow(G) -> int:
    """Add the reverse of every one-way edge that cyclists may ride both ways.

    Runs on the UNSIMPLIFIED graph, where each edge is one OSM way segment and
    still carries its own tags. Returns the number of edges added.
    """
    added = 0
    for u, v, k, d in list(G.edges(keys=True, data=True)):
        if not d.get("oneway"):
            continue
        ob = str(d.get("oneway:bicycle", "")).strip().lower()
        cw = str(d.get("cycleway", "")).strip().lower()
        if ob != "no" and not cw.startswith("opposite"):
            continue
        if G.has_edge(v, u):
            continue
        rev = dict(d)
        rev["oneway"] = False
        rev["reversed"] = not bool(d.get("reversed", False))
        rev.pop("geometry", None)
        G.add_edge(v, u, **rev)
        added += 1
    return added


def prune_components(G, min_nodes: int) -> Tuple[object, List[dict]]:
    """Keep each sizeable piece of the network, drop small fragments.

    For every weakly connected component with at least ``min_nodes`` nodes the
    largest strongly connected part is kept, so that within a kept piece every
    node can reach every other (on a walk network, which is bidirectional, the
    two coincide). Returns the pruned graph and a size report, largest first.
    """
    keep = set()
    report = []
    for wcc in nx.weakly_connected_components(G):
        if len(wcc) < min_nodes:
            continue
        sub = G.subgraph(wcc)
        scc = max(nx.strongly_connected_components(sub), key=len)
        keep |= scc
        report.append({"wcc_nodes": len(wcc), "kept_nodes": len(scc)})
    report.sort(key=lambda r: -r["kept_nodes"])
    return G.subgraph(keep).copy(), report


def _download_graph(core_polygon, mode: str, graphml_path: str, log=print):
    """Download, clean and cache one mode's network (GraphML)."""
    if os.path.exists(graphml_path):
        return ox.load_graphml(graphml_path)

    from .netfix import configure_overpass

    ap = params.ACTIVE_NETWORK_PARAMS
    overpass_url = configure_overpass(refresh=True)
    if overpass_url:
        log(f"  overpass endpoint: {overpass_url} "
            f"(rate limiter {'on' if ox.settings.overpass_rate_limit else 'off'})")

    ox.settings.use_cache = False
    polygon = download_polygon(core_polygon)
    saved_tags = list(ox.settings.useful_tags_way)
    try:
        if mode == "bike":
            # Carry the tags contraflow detection reads through graph creation.
            ox.settings.useful_tags_way = saved_tags + [
                t for t in ("oneway:bicycle", "cycleway") if t not in saved_tags]
            from osmnx import _overpass

            filters = [_overpass._get_network_filter("bike"), *ap["bike_extra_filters"]]
            G = ox.graph_from_polygon(polygon, network_type="bike", custom_filter=filters,
                                      simplify=False, retain_all=True)
            if ap["bike_contraflow"]:
                n = _add_contraflow(G)
                log(f"  bike: {n:,} contraflow edges added")
        elif mode == "walk":
            # network_type="walk" keeps the graph bidirectional; the custom
            # filter only changes which ways are fetched.
            G = ox.graph_from_polygon(polygon, network_type="walk",
                                      custom_filter=[walk_filter(), *ap["walk_extra_filters"]],
                                      simplify=False, retain_all=True)
        else:
            raise ValueError(f"unknown active mode {mode!r}")
    finally:
        ox.settings.useful_tags_way = saved_tags

    G = ox.simplify_graph(G)
    G, report = prune_components(G, int(ap["min_component_nodes"]))
    G.graph["component_report"] = json.dumps(report)
    G.graph["active_spec"] = json.dumps(_spec(mode), sort_keys=True)
    os.makedirs(os.path.dirname(graphml_path), exist_ok=True)
    ox.save_graphml(G, graphml_path)
    return G


def build_active_network(metro: str, mode: str, drive_net: RoadNetwork, *,
                         cache_dir: Optional[str] = None, use_warm: bool = True,
                         log=print) -> RoadNetwork:
    """Build (or load) the ``mode`` network for ``metro``, POIs baked in.

    ``drive_net`` is the metro's finished drive network. It supplies the core
    polygon and the POI selection, so both networks hold the same venues.
    """
    cache_dir = cache_dir or params.GEO_PARAMS["cache_dir"]
    warm = _warm_path(cache_dir, metro, mode)
    poi_fp = getattr(drive_net, "poi_fingerprint", "")
    spec_hash = _spec_hash(mode)

    if use_warm and os.path.exists(warm):
        with open(warm, "rb") as f:
            net = pickle.load(f)
        if (getattr(net, "warm_version", 0) == ACTIVE_WARM_VERSION
                and getattr(net, "poi_fingerprint", None) == poi_fp
                and getattr(net, "active_spec_hash", None) == spec_hash):
            net.cache_dir = cache_dir
            net.metro = metro
            net.core_polygon = drive_net.core_polygon
            routing_matrix.attach(net, metro, cache_dir, poi_fp, mode=mode)
            return net
        log(f"  {metro}/{mode}: warm pickle is stale (POIs or network spec "
            f"changed); rebuilding")

    if drive_net.core_polygon is None:
        raise ValueError(f"{metro}: drive network has no core polygon")
    graph = _download_graph(drive_net.core_polygon, mode,
                            _graphml_path(cache_dir, metro, mode), log=log)
    net = RoadNetwork(f"{metro}_{mode}", graph=graph, network_type=mode,
                      cache_dir=cache_dir, timed=False)
    net.core_polygon = drive_net.core_polygon
    net.metro = metro

    poi_latlon = getattr(drive_net, "poi_latlon", {}) or {}
    if poi_latlon:
        net.add_pois_as_nodes([(pid, lat, lon) for pid, (lat, lon) in poi_latlon.items()])
    net.poi_latlon = dict(poi_latlon)
    net.poi_fingerprint = poi_fp
    net.poi_cap = getattr(drive_net, "poi_cap", None)
    net.warm_version = ACTIVE_WARM_VERSION
    net.active_spec_hash = spec_hash

    os.makedirs(os.path.dirname(warm), exist_ok=True)
    with open(warm, "wb") as f:
        pickle.dump(net, f, protocol=_PICKLE_PROTOCOL)
    routing_matrix.attach(net, metro, cache_dir, poi_fp, mode=mode)
    return net


def attach_active_networks(drive_net: RoadNetwork, metro: str, *,
                           cache_dir: Optional[str] = None, log=print) -> None:
    """Attach every mode in ``ACTIVE_NETWORK_PARAMS['modes']`` to ``drive_net``."""
    nets: Dict[str, RoadNetwork] = {}
    for mode in params.ACTIVE_NETWORK_PARAMS["modes"]:
        nets[mode] = build_active_network(metro, mode, drive_net,
                                          cache_dir=cache_dir, log=log)
    drive_net.mode_networks = nets


def attach_bbox_active_networks(drive_net: RoadNetwork, bbox, *,
                                cache_dir: Optional[str] = None, log=print) -> None:
    """Walk and bike networks for a legacy single-area preset (a bbox).

    The whole box is the study area there (the drive network has no core
    polygon, so every agent lives "in the core"). No POIs are baked in and no
    matrices are built — the Simulation inserts its POIs at run time and these
    areas are small enough for live routing.
    """
    from shapely.geometry import box

    cache_dir = cache_dir or params.GEO_PARAMS["cache_dir"]
    west, south, east, north = bbox
    area = box(west, south, east, north)
    nets: Dict[str, RoadNetwork] = {}
    for mode in params.ACTIVE_NETWORK_PARAMS["modes"]:
        path = _graphml_path(cache_dir, drive_net.name, mode)
        graph = _download_graph(area, mode, path, log=log)
        net = RoadNetwork(f"{drive_net.name}_{mode}", graph=graph, network_type=mode,
                          cache_dir=cache_dir, timed=False)
        nets[mode] = net
    drive_net.mode_networks = nets


def ensure_active_matrices(drive_net: RoadNetwork, metro: str, *,
                           cache_dir: Optional[str] = None,
                           workers: Optional[int] = None, rebuild: bool = False,
                           log=print) -> None:
    """Build any missing (or stale) walk/bike routing matrices for ``metro``."""
    cache_dir = cache_dir or params.GEO_PARAMS["cache_dir"]
    if not drive_net.mode_networks:
        attach_active_networks(drive_net, metro, cache_dir=cache_dir, log=log)
    poi_fp = getattr(drive_net, "poi_fingerprint", "")
    for mode, net in drive_net.mode_networks.items():
        if net.has_routing_matrix and not rebuild:
            meta = routing_matrix.read_meta(cache_dir, metro, mode) or {}
            log(f"  {metro}/{mode}: routing matrices already current "
                f"({meta.get('universe', 0):,} endpoint nodes)")
            continue
        routing_matrix.build(net, metro, cache_dir=cache_dir, poi_fingerprint=poi_fp,
                             workers=workers, log=log, mode=mode, source=drive_net)
        routing_matrix.attach(net, metro, cache_dir, poi_fp, mode=mode)

"""Two-layer metro road networks: full-detail core + arterial shell.

Builds ONE :class:`welfare_rs.geo.RoadNetwork` per metro, assembled from two
OSM downloads and thought of as two abstract layers:

* **core** — the principal city polygon (``params.METRO_PARAMS`` →
  ``core_places``), full ``drive`` network. POIs and work locations live only
  here; leisure routing stays street-level.
* **shell** — the union of the metro counties that compose 90% of the workers
  commuting into the core (``datasource.select_counties``), reduced to the
  arterial skeleton (motorway/trunk/primary/secondary). Homes live anywhere;
  commutes ride arterials. Dropping residential capillaries outside the core
  is the standard MPO travel-model abstraction and keeps metro graphs ~5–10×
  smaller than full drive detail.

Both pieces are downloaded **unsimplified** so they share raw OSM node ids,
composed, then simplified once — a clean weld with no duplicate parallel
representations at the boundary. The result is cached three ways:

* geocoded boundary polygons  → ``<cache>/boundaries/*.json`` (GeoJSON-ish)
* the composed graph          → ``<cache>/networks/<metro>_metro.graphml``
* the finished RoadNetwork    → ``<cache>/warmed/<metro>_metro.pkl`` (fast
  loads for worker processes; GraphML parsing is minutes at metro scale)

First build per metro downloads from Overpass and can take a while — use
``scripts/warm_metros.py`` to prefetch all metros once.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
from typing import Dict, Optional

import networkx as nx
import osmnx as ox

from . import params
from .datasource import DataSource, get_datasource, select_counties
from .geo import RoadNetwork

_PICKLE_PROTOCOL = 4


# ── Boundary polygons (geocoded once, cached on disk) ────────────────────────

def _boundaries_dir(cache_dir: str) -> str:
    path = os.path.join(cache_dir, "boundaries")
    os.makedirs(path, exist_ok=True)
    return path


def geocode_polygon(query: str, cache_dir: Optional[str] = None):
    """Geocode a place query to a shapely (Multi)Polygon via Nominatim,
    cached on disk so each boundary is fetched exactly once per machine."""
    from shapely.geometry import mapping, shape

    cache_dir = cache_dir or params.GEO_PARAMS["cache_dir"]
    slug = hashlib.md5(query.strip().lower().encode("utf-8")).hexdigest()[:16]
    path = os.path.join(_boundaries_dir(cache_dir), f"{slug}.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return shape(json.load(f)["geometry"])

    gdf = ox.geocoder.geocode_to_gdf(query)
    try:
        geom = gdf.geometry.union_all()
    except AttributeError:  # geopandas < 1.0
        geom = gdf.geometry.unary_union
    if geom.is_empty:
        raise ValueError(f"Nominatim returned an empty boundary for {query!r}")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"query": query, "geometry": mapping(geom)}, f)
    return geom


def _union(geoms):
    from shapely.ops import unary_union

    return unary_union(list(geoms))


# ── Metro network assembly ───────────────────────────────────────────────────

def _warm_path(cache_dir: str, metro: str) -> str:
    path = os.path.join(cache_dir, "warmed")
    os.makedirs(path, exist_ok=True)
    return os.path.join(path, f"{metro}_metro.pkl")


def _compose_metro_graph(core_polygon, metro_polygon, graphml_path: str):
    """Download core (full drive) + shell (arterials), weld, simplify, SCC."""
    if os.path.exists(graphml_path):
        return ox.load_graphml(graphml_path)

    ox.settings.use_cache = False
    # Unsimplified downloads share raw OSM node ids wherever the same road
    # appears in both queries (arterials inside the core), so compose() welds
    # the layers exactly; simplifying afterwards collapses degree-2 chains
    # once, consistently. retain_all keeps disjoint pieces (e.g. the two core
    # cities of a multi-city core) alive until the SCC cut on the whole.
    core_g = ox.graph_from_polygon(
        core_polygon, network_type="drive", simplify=False, retain_all=True
    )
    shell_g = ox.graph_from_polygon(
        metro_polygon,
        custom_filter=params.METRO_PARAMS["arterial_filter"],
        simplify=False,
        retain_all=True,
    )
    graph = nx.compose(shell_g, core_g)  # core attrs win on shared edges
    graph = ox.simplify_graph(graph)
    graph = ox.truncate.largest_component(graph, strongly=True)
    os.makedirs(os.path.dirname(graphml_path), exist_ok=True)
    ox.save_graphml(graph, graphml_path)
    return graph


def build_metro_network(
    metro: str,
    *,
    datasource: Optional[DataSource] = None,
    cache_dir: Optional[str] = None,
    use_warm: bool = True,
) -> RoadNetwork:
    """Build (or load from cache) the two-layer network for a metro key.

    The datasource decides which counties are in (90% inbound-worker rule)
    and provides their weights; geometry comes from Nominatim + Overpass on
    first build and pure disk caches afterwards.
    """
    metros = params.METRO_PARAMS["metros"]
    if metro not in metros:
        raise KeyError(
            f"Unknown metro {metro!r}; known: {', '.join(metros)}"
        )
    cache_dir = cache_dir or params.GEO_PARAMS["cache_dir"]

    warm = _warm_path(cache_dir, metro)
    if use_warm and os.path.exists(warm):
        with open(warm, "rb") as f:
            net = pickle.load(f)
        net.cache_dir = cache_dir
        return net

    ds = datasource or get_datasource()
    counties = select_counties(ds.county_flows(metro))

    core_polygon = _union(
        geocode_polygon(q, cache_dir) for q in metros[metro]["core_places"]
    )
    county_polygons: Dict[str, object] = {
        c.fips: geocode_polygon(c.geocode, cache_dir) for c in counties
    }
    metro_polygon = _union([core_polygon, *county_polygons.values()])

    graphml_path = os.path.join(cache_dir, "networks", f"{metro}_metro.graphml")
    graph = _compose_metro_graph(core_polygon, metro_polygon, graphml_path)

    net = RoadNetwork(f"{metro}_metro", graph=graph, cache_dir=cache_dir)
    net.attach_layers(
        core_polygon,
        county_polygons,
        {c.fips: {"name": c.name, "workers": c.workers, "core_share": c.core_share}
         for c in counties},
    )
    with open(warm, "wb") as f:
        pickle.dump(net, f, protocol=_PICKLE_PROTOCOL)
    return net


def is_metro(city_key: str) -> bool:
    """True when a city key names a two-layer metro (vs a legacy bbox preset)."""
    return city_key in params.METRO_PARAMS["metros"]


def build_network(city_key: str, **kwargs) -> RoadNetwork:
    """Dispatch: metro keys build two-layer networks, legacy preset keys fall
    back to :func:`welfare_rs.geo.build_road_network`."""
    if is_metro(city_key):
        return build_metro_network(city_key, **kwargs)
    from .geo import build_road_network

    return build_road_network(city_key, cache_dir=kwargs.get("cache_dir"))

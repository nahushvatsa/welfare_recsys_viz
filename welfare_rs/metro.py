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


def _union(geoms, weld_m: float = 0.0):
    """Union of polygons, optionally welded across hairline gaps.

    Counties are geocoded one at a time, so two that share a real border can
    come back as polygons that merely *touch* — Nominatim's Cook County IL and
    Lake County IN meet at a single POINT, with 0.000 km of shared edge.
    ``unary_union`` then returns a MultiPolygon, osmnx downloads the parts as
    disconnected pieces, and ``largest_component(strongly=True)`` deletes the
    smaller one. That is how every road in Lake County Indiana disappeared from
    the Chicago graph, leaving its 19,576 commuters to snap up to 30 km west
    into Illinois — the same failure as Seattle's Clark County, reached through
    geometry rather than distance.

    Buffering each polygon by a few tens of metres before the union closes
    those gaps. It also makes the download area marginally larger than the
    county union, which is harmless and mildly useful: arterials at the rim are
    truncated a little further out, so fewer of them become dead-end stubs that
    the strongly-connected cut discards.
    """
    from shapely.ops import unary_union

    geoms = list(geoms)
    if weld_m:
        deg = weld_m / 111_000.0  # metres -> degrees, near enough at these lats
        geoms = [g.buffer(deg) for g in geoms]
    return unary_union(geoms)


# Wide enough to close geocoder seams, far too small to pull in a neighbouring
# town. Chicago's union goes from 2 parts to 1 at 10 m; 50 m is margin.
_WELD_M = 50.0


# ── Metro network assembly ───────────────────────────────────────────────────

def excluded_counties(metro: str) -> set:
    """County FIPS the coverage rule selects but the graph cannot serve.

    ``select_counties`` ranks purely by inbound worker volume and never asks
    whether a county is anywhere near the metro, so a high-volume outlier can
    be selected while the counties between it and the core are not. Its roads
    then form an island that the strongly-connected-component cut deletes,
    leaving its homes to snap tens of kilometres to the nearest survivor and be
    charged the distance as travel time. Listing such a county here removes it
    from both the download and the commute pairs.
    """
    meta = params.METRO_PARAMS["metros"].get(metro, {})
    return set(meta.get("exclude_counties", ()))


def _warm_path(cache_dir: str, metro: str) -> str:
    path = os.path.join(cache_dir, "warmed")
    os.makedirs(path, exist_ok=True)
    return os.path.join(path, f"{metro}_metro.pkl")


def _compose_metro_graph(core_polygon, metro_polygon, graphml_path: str):
    """Download core (full drive) + shell (arterials), weld, simplify, SCC."""
    if os.path.exists(graphml_path):
        return ox.load_graphml(graphml_path)

    # Endpoint is measured, not assumed — down to the backend IP — and osmnx's
    # rate limiter is enabled only if the endpoint actually meters. See
    # welfare_rs.netfix for the three separate failures this covers.
    # refresh=True: re-probe for THIS metro rather than trusting a choice made
    # before the previous metro, which may have gone dead since.
    from .netfix import configure_overpass

    overpass_url = configure_overpass(refresh=True)
    if overpass_url:
        print(f"  overpass endpoint: {overpass_url} "
              f"(rate limiter {'on' if ox.settings.overpass_rate_limit else 'off'})",
              flush=True)

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
        # Commute pairs are deliberately not pickled (millions of rows); each
        # process re-attaches them, restricted to the counties this graph
        # actually covers — county_meta is exactly that set.
        _attach_commutes(net, metro, datasource, list(net.county_meta or {}))
        return net

    ds = datasource or get_datasource()
    drop = excluded_counties(metro)
    counties = [c for c in select_counties(ds.county_flows(metro))
                if c.fips not in drop]

    core_polygon = _union(
        geocode_polygon(q, cache_dir) for q in metros[metro]["core_places"]
    )
    county_polygons: Dict[str, object] = {
        c.fips: geocode_polygon(c.geocode, cache_dir) for c in counties
    }
    # Welded: see _union. Genuine offshore islands (Catalina, the Farallones,
    # the Channel Islands) stay separate at this buffer and are meant to — the
    # strongly-connected cut drops them because you cannot drive there.
    metro_polygon = _union([core_polygon, *county_polygons.values()],
                           weld_m=_WELD_M)

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
    _attach_commutes(net, metro, ds, [c.fips for c in counties])
    return net


def _attach_commutes(net: RoadNetwork, metro: str,
                     datasource: Optional[DataSource], counties) -> None:
    """Attach real block-level commutes, if the datasource has them.

    Restricted to ``counties`` — the ones built into this graph — because a
    home block outside the graph would snap to whatever node happens to sit on
    the graph's boundary. Failure is non-fatal: without pairs the network keeps
    its county-weighted home sampler and a uniform workplace in the core.
    """
    ds = datasource or get_datasource()
    # Filtered here as well as at selection time so an ALREADY-WARMED pickle is
    # corrected on load: county_meta was baked in before the exclusion existed,
    # and re-downloading a metro graph to drop a county it never usefully
    # contained would be a waste.
    drop = excluded_counties(metro)
    counties = [c for c in counties if c not in drop]
    try:
        pairs = ds.commute_pairs(metro, counties)
    except Exception:
        pairs = None
    net.attach_commutes(pairs)


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

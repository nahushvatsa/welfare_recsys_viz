"""Server-side geometry helpers for the deck.gl frontend.

The backend owns the full simulation run; the browser only ever pulls what it
needs to draw the current time window inside the current map viewport. These
pure functions turn a completed :class:`welfare_rs.Simulation` day into:

* **timelines** — per-agent moving (trip) and stationary (activity) segments.
  Each route is simplified to a ~11 m tolerance and held as a float32 array, so
  an animated dot still tracks the real street geometry while the run stays
  bounded in memory (routes dominate it: they scale as agents x days x seeds).
* **windowed trips / stays** — the trips/stays falling in a requested time
  window and map bounding box. The client animates along these keyframes.
* **on-demand geometry** — one trip's polyline, fetched when a dot is inspected.

Locations are ``(lat, lon)``; deck.gl wants ``[lon, lat]`` so the windowed
outputs emit that order. No web/visualization framework is imported here — these
are plain data transforms reused by the API layer.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from welfare_rs.geo import haversine_km

LatLon = Tuple[float, float]
BBox = Tuple[float, float, float, float]  # (south, west, north, east)

DAY_MINUTES = 1440

# RGB colours shared with the frontend legend.
MODE_COLOR = {
    "car": [230, 80, 60],
    "walk": [80, 170, 90],
    "bike": [240, 180, 40],
    "transit": [70, 130, 220],
}
STATE_COLOR = {
    "home": [80, 170, 90],
    "work": [240, 150, 40],
    "leisure": [160, 90, 210],
    "in_transit": [230, 80, 60],
}

# Douglas-Peucker tolerance for stored routes, in degrees (~1e-4 deg ~ 11 m).
#
# Routes are simplified by METRIC TOLERANCE rather than to a fixed point count.
# That distinction is the whole point: a fixed count (this module used to send
# 12 evenly-spaced points) makes a dot visibly cut corners, because a 25 km
# commute becomes 11 straight hops. A tolerance keeps every corner and curve
# while collapsing a straight freeway run to its two endpoints, so the dot
# tracks the real road at any usable zoom — and the result is ~100x smaller
# than the raw polyline it replaces.
PATH_TOL_DEG = float(os.environ.get("WELFARE_RS_PATH_TOL_DEG", "1e-4"))

_EARTH_RADIUS_KM = 6371.0088


# ── Geometry math ─────────────────────────────────────────────────────────────

def _simplify_path(path: Sequence[LatLon]) -> np.ndarray:
    """Simplify a (lat, lon) polyline and return it as a float32 (N, 2) array.

    float32 gives ~1 m positional resolution at these latitudes — far finer
    than the simplification tolerance — at 8 bytes per point instead of the
    ~100 a Python tuple of floats costs.
    """
    if len(path) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    if len(path) < 3:
        return np.asarray(path, dtype=np.float32).reshape(-1, 2)
    from shapely.geometry import LineString

    line = LineString([(lon, lat) for lat, lon in path])
    simple = line.simplify(PATH_TOL_DEG, preserve_topology=False)
    return np.array([(y, x) for x, y in simple.coords], dtype=np.float32)


def _cumulative_km(path) -> np.ndarray:
    """Cumulative great-circle distance (km) along a polyline, as float32."""
    p = np.asarray(path, dtype=np.float64).reshape(-1, 2)
    if len(p) < 2:
        return np.zeros(len(p), dtype=np.float32)
    lat1, lon1 = np.radians(p[:-1, 0]), np.radians(p[:-1, 1])
    lat2, lon2 = np.radians(p[1:, 0]), np.radians(p[1:, 1])
    h = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    seg = 2.0 * _EARTH_RADIUS_KM * np.arcsin(np.sqrt(h))
    out = np.zeros(len(p), dtype=np.float64)
    np.cumsum(seg, out=out[1:])
    return out.astype(np.float32)


def _position_at(path, cum, frac: float) -> LatLon:
    """Interpolate a position along ``path`` at distance fraction ``frac``."""
    n = len(path)
    if n == 0:
        return (0.0, 0.0)
    if n == 1:
        return (float(path[0][0]), float(path[0][1]))
    target = max(0.0, min(1.0, frac)) * float(cum[-1])
    i = int(np.searchsorted(cum, target, side="left"))
    if i <= 0:
        return (float(path[0][0]), float(path[0][1]))
    if i >= n:
        return (float(path[-1][0]), float(path[-1][1]))
    a, b = float(cum[i - 1]), float(cum[i])
    f = 0.0 if b <= a else (target - a) / (b - a)
    lat = float(path[i - 1][0]) + (float(path[i][0]) - float(path[i - 1][0])) * f
    lon = float(path[i - 1][1]) + (float(path[i][1]) - float(path[i - 1][1])) * f
    return (lat, lon)


# ── Timelines (per simulated day) ──────────────────────────────────────────────

def build_timelines(sim) -> Dict[int, dict]:
    """Per-agent timeline of moving (trip) and stationary (activity) segments
    for the simulation's *current* day.

    Reconstructs each agent's day from its realised trips: between trips the
    agent waits at the previous destination, with the state there being the
    purpose of the trip that brought it (home/work/leisure). ``path`` is the
    simplified route (float32 lat/lon) and ``cum`` its cumulative distances.
    """
    net = sim.road_network
    timelines: Dict[int, dict] = {}
    for agent in sim.agents:
        trips = sorted(agent.trips, key=lambda t: t.depart_time)
        moves: List[dict] = []
        stays: List[Tuple[int, int, LatLon, str]] = []
        prev_end = 0
        prev_loc: LatLon = agent.home
        prev_state = "home"
        for tr in trips:
            if tr.depart_time > prev_end:
                stays.append((prev_end, tr.depart_time, prev_loc, prev_state))
            path = _simplify_path(net.route_geometry_latlon(tr.origin, tr.destination))
            moves.append(
                {
                    "t0": tr.depart_time,
                    "t1": tr.arrival_time,
                    "path": path,
                    "cum": _cumulative_km(path),
                    "mode": tr.mode,
                    "purpose": tr.purpose,
                }
            )
            prev_end = tr.arrival_time
            prev_loc = tr.destination
            prev_state = tr.purpose
        stays.append((prev_end, DAY_MINUTES, prev_loc, prev_state))
        timelines[agent.id] = {"moves": moves, "stays": stays}
    return timelines


def merge_day_timelines(per_day: Sequence[Dict[int, dict]]) -> Dict[int, dict]:
    """Concatenate per-day timelines into one multi-day timeline.

    Day ``d``'s times are shifted by ``d * 1440`` minutes so the days lay
    end-to-end, and every move is tagged with a stable ``trip_id``
    (``"<agent>-<day>-<index>"``) the client uses to request full geometry.
    """
    merged: Dict[int, dict] = {}
    for day_index, tl_day in enumerate(per_day):
        off = day_index * DAY_MINUTES
        for aid, tl in tl_day.items():
            m = merged.setdefault(aid, {"moves": [], "stays": []})
            for i, mv in enumerate(tl["moves"]):
                m["moves"].append(
                    {
                        "trip_id": f"{aid}-{day_index}-{i}",
                        "t0": mv["t0"] + off,
                        "t1": mv["t1"] + off,
                        "path": mv["path"],
                        "cum": mv["cum"],
                        "mode": mv["mode"],
                        "purpose": mv["purpose"],
                    }
                )
            for (s0, s1, loc, state) in tl["stays"]:
                m["stays"].append((s0 + off, s1 + off, loc, state))
    return merged


def build_trip_index(merged: Dict[int, dict]) -> Dict[str, dict]:
    """Map ``trip_id`` -> move record, for O(1) on-demand geometry lookups."""
    return {mv["trip_id"]: mv for tl in merged.values() for mv in tl["moves"]}


def time_span(merged: Dict[int, dict]) -> int:
    """Largest timestamp across all moves/stays (minutes); 0 if empty."""
    hi = 0
    for tl in merged.values():
        for mv in tl["moves"]:
            hi = max(hi, mv["t1"])
        for (_s0, s1, _loc, _st) in tl["stays"]:
            hi = max(hi, s1)
    return hi


# ── Bounding-box helpers ────────────────────────────────────────────────────

def _in_bbox(lat: float, lon: float, bbox: BBox) -> bool:
    south, west, north, east = bbox
    return south <= lat <= north and west <= lon <= east


def _bbox_overlap(path, bbox: BBox) -> bool:
    """True if the polyline's extent intersects the viewport bbox (so a trip
    passing through the view is kept even when its endpoints are off-screen)."""
    south, west, north, east = bbox
    p = np.asarray(path, dtype=np.float64).reshape(-1, 2)
    if p.size == 0:
        return False
    if p[:, 0].max() < south or p[:, 0].min() > north:
        return False
    if p[:, 1].max() < west or p[:, 1].min() > east:
        return False
    return True


# ── Windowed payloads (time + viewport culled) ─────────────────────────────────

def _trip_record(mv: dict) -> dict:
    """Compact, downsampled position keyframes for one move (deck.gl order)."""
    path, cum = mv["path"], mv["cum"]
    total = float(cum[-1]) or 1.0
    t0, t1 = mv["t0"], mv["t1"]
    timestamps = [round(t0 + (t1 - t0) * (float(c) / total), 1) for c in cum]
    return {
        "agent_id": int(mv["trip_id"].split("-")[0]),
        "trip_id": mv["trip_id"],
        "path": [[round(float(lon), 5), round(float(lat), 5)] for lat, lon in path],
        "timestamps": timestamps,
        "mode": mv["mode"],
        "color": MODE_COLOR.get(mv["mode"], [200, 200, 200]),
    }


def window_trips(
    merged: Dict[int, dict], t0: float, t1: float, bbox: Optional[BBox] = None
) -> List[dict]:
    """Downsampled trip keyframes for moves overlapping ``[t0, t1]`` (and the
    optional viewport ``bbox``). The client interpolates dots along these."""
    out: List[dict] = []
    for tl in merged.values():
        for mv in tl["moves"]:
            if mv["t1"] < t0 or mv["t0"] > t1:
                continue
            if bbox is not None and not _bbox_overlap(mv["path"], bbox):
                continue
            out.append(_trip_record(mv))
    return out


def window_stays(
    merged: Dict[int, dict], t0: float, t1: float, bbox: Optional[BBox] = None
) -> List[dict]:
    """Stationary segments overlapping ``[t0, t1]`` (and the optional bbox)."""
    out: List[dict] = []
    for aid, tl in merged.items():
        for (s0, s1, loc, state) in tl["stays"]:
            if s1 < t0 or s0 > t1:
                continue
            lat, lon = loc
            if bbox is not None and not _in_bbox(lat, lon, bbox):
                continue
            out.append(
                {
                    "agent_id": aid,
                    "t0": s0,
                    "t1": s1,
                    "lon": round(lon, 5),
                    "lat": round(lat, 5),
                    "state": state,
                    "color": STATE_COLOR.get(state, [150, 150, 150]),
                }
            )
    return out


def trip_geometry(mv: dict) -> dict:
    """Full street polyline + per-vertex timestamps for one move (on demand).

    Returns the complete route (not downsampled) so the client can draw a
    faithful TripsLayer trail for the inspected trip.
    """
    cum = mv["cum"]
    # float() on both sides, as in _trip_record: `cum` is a float32 array, and
    # without the casts every timestamp comes back a numpy.float32, which the
    # JSON serializer refuses — a 500 on every click-to-inspect.
    total = float(cum[-1]) or 1.0
    t0, t1 = mv["t0"], mv["t1"]
    return {
        "trip_id": mv["trip_id"],
        "agent_id": int(mv["trip_id"].split("-")[0]),
        "path": [[round(float(lon), 6), round(float(lat), 6)] for lat, lon in mv["path"]],
        "timestamps": [round(t0 + (t1 - t0) * (float(c) / total), 1) for c in cum],
        "mode": mv["mode"],
        "purpose": mv["purpose"],
        "color": MODE_COLOR.get(mv["mode"], [200, 200, 200]),
    }


# ── Exact-instant positions (optional; full-fidelity interpolation) ─────────────

def _marker_at(tl: dict, t: float):
    """(lat, lon, state, mode) for one agent at time t, or None."""
    for mv in tl["moves"]:
        if mv["t0"] <= t <= mv["t1"]:
            span = max(1, mv["t1"] - mv["t0"])
            pos = _position_at(mv["path"], mv["cum"], (t - mv["t0"]) / span)
            return pos[0], pos[1], "in_transit", mv["mode"]
    for (s0, s1, loc, state) in tl["stays"]:
        if s0 <= t <= s1:
            return loc[0], loc[1], state, None
    return None


def positions_at(merged: Dict[int, dict], t: float, bbox: Optional[BBox] = None) -> List[dict]:
    """Every agent's compact marker at an exact instant ``t`` (minutes), with
    in-transit agents interpolated along their real route. Viewport-culled."""
    out: List[dict] = []
    for aid, tl in merged.items():
        m = _marker_at(tl, t)
        if m is None:
            continue
        lat, lon, state, mode = m
        if bbox is not None and not _in_bbox(lat, lon, bbox):
            continue
        out.append(
            {
                "agent_id": aid,
                "lon": round(lon, 5),
                "lat": round(lat, 5),
                "state": state,
                "mode": mode or "",
                "color": STATE_COLOR.get(state, [150, 150, 150]),
            }
        )
    return out


# ── POIs ───────────────────────────────────────────────────────────────────

def pois_in_bbox(pois: Sequence[dict], bbox: Optional[BBox] = None, limit: int = 4000) -> List[dict]:
    """POIs inside the viewport bbox (all of them when bbox is None), capped to
    ``limit`` so dense areas stay responsive."""
    if bbox is None:
        return list(pois[:limit])
    out = []
    for p in pois:
        if _in_bbox(p["lat"], p["lon"], bbox):
            out.append(p)
            if len(out) >= limit:
                break
    return out

"""Server-side geometry helpers for the deck.gl frontend.

The backend owns the full simulation run; the browser only ever pulls what it
needs to draw the current time window inside the current map viewport. These
pure functions turn a completed :class:`welfare_rs.Simulation` day into:

* **timelines** — per-agent moving (trip) and stationary (activity) segments,
  retaining the *full* street-route polyline so positions can be interpolated at
  any instant with full fidelity (the MATSim/SUMO look).
* **windowed trips / stays** — compact, *downsampled* position keyframes for the
  trips/stays that fall in a requested time window and map bounding box. This is
  the small default payload the client animates client-side.
* **on-demand geometry** — the full street polyline for a single trip, fetched
  only when a dot is inspected.

Locations are ``(lat, lon)``; deck.gl wants ``[lon, lat]`` so the windowed
outputs emit that order. No web/visualization framework is imported here — these
are plain data transforms reused by the API layer.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

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

MAX_PATH_POINTS = 12  # downsample route polylines to bound the windowed payload


# ── Geometry math ─────────────────────────────────────────────────────────────

def _cumulative_km(path: List[LatLon]) -> List[float]:
    """Cumulative great-circle distance (km) along a polyline."""
    cum = [0.0]
    for i in range(1, len(path)):
        cum.append(cum[-1] + haversine_km(path[i - 1], path[i]))
    return cum


def _position_at(path: List[LatLon], cum: List[float], frac: float) -> LatLon:
    """Interpolate a position along ``path`` at distance fraction ``frac``."""
    if len(path) == 1:
        return path[0]
    target = max(0.0, min(1.0, frac)) * cum[-1]
    for i in range(1, len(path)):
        if cum[i] >= target:
            seg = cum[i] - cum[i - 1]
            f = 0.0 if seg <= 0 else (target - cum[i - 1]) / seg
            lat = path[i - 1][0] + (path[i][0] - path[i - 1][0]) * f
            lon = path[i - 1][1] + (path[i][1] - path[i - 1][1]) * f
            return (lat, lon)
    return path[-1]


def _downsample(path: List[LatLon], k: int) -> List[LatLon]:
    """Keep ~k evenly-spaced points (including both endpoints) from a polyline."""
    n = len(path)
    if n <= k:
        return path
    step = (n - 1) / (k - 1)
    return [path[round(i * step)] for i in range(k)]


# ── Timelines (per simulated day) ──────────────────────────────────────────────

def build_timelines(sim) -> Dict[int, dict]:
    """Per-agent timeline of moving (trip) and stationary (activity) segments
    for the simulation's *current* day.

    Reconstructs each agent's day from its realised trips: between trips the
    agent waits at the previous destination, with the state there being the
    purpose of the trip that brought it (home/work/leisure). The full route
    polyline (``path``) and its cumulative distances (``cum``) are retained.
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
            path = net.route_geometry_latlon(tr.origin, tr.destination)
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


def _bbox_overlap(path: List[LatLon], bbox: BBox) -> bool:
    """True if the polyline's extent intersects the viewport bbox (so a trip
    passing through the view is kept even when its endpoints are off-screen)."""
    south, west, north, east = bbox
    lats = [p[0] for p in path]
    lons = [p[1] for p in path]
    if max(lats) < south or min(lats) > north:
        return False
    if max(lons) < west or min(lons) > east:
        return False
    return True


# ── Windowed payloads (time + viewport culled) ─────────────────────────────────

def _trip_record(mv: dict) -> dict:
    """Compact, downsampled position keyframes for one move (deck.gl order)."""
    path = _downsample(mv["path"], MAX_PATH_POINTS)
    cum = _cumulative_km(path)
    total = cum[-1] or 1.0
    t0, t1 = mv["t0"], mv["t1"]
    timestamps = [round(t0 + (t1 - t0) * (c / total), 1) for c in cum]
    return {
        "agent_id": int(mv["trip_id"].split("-")[0]),
        "trip_id": mv["trip_id"],
        "path": [[round(lon, 5), round(lat, 5)] for (lat, lon) in path],
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
    total = cum[-1] or 1.0
    t0, t1 = mv["t0"], mv["t1"]
    return {
        "trip_id": mv["trip_id"],
        "agent_id": int(mv["trip_id"].split("-")[0]),
        "path": [[round(lon, 6), round(lat, 6)] for (lat, lon) in mv["path"]],
        "timestamps": [round(t0 + (t1 - t0) * (c / total), 1) for c in cum],
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

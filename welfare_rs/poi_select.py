"""Deterministic POI catalog selection, shared by warming and simulation.

The selection has to be byte-for-byte reproducible across processes and across
machines, for two reasons:

* POIs are inserted into the road graph as mid-block nodes, so the *set* of
  selected POIs determines the graph's node ids.
* Those node ids index the precomputed routing matrices
  (:mod:`welfare_rs.routing_matrix`). A different selection silently shifts
  every row and column.

Determinism is not automatic here. The per-category cap draws a random sample
*by position*, so it is only stable if the candidate list is in a stable order —
and ``SELECT ... FROM metro_pois`` makes no ordering promise. Candidates are
therefore sorted by a total key before any sampling happens, and categories are
walked in sorted order so the resulting node-id assignment is fixed too.
"""

from __future__ import annotations

import hashlib
import random
from typing import Dict, List, Optional, Sequence, Tuple

from . import params

# Fixed, deliberately unrelated to the simulation seed: the POI set must be
# identical across seeds and treatments or matched comparisons are confounded.
CAP_SEED = 20240608

# (place_id, name, category, lat, lon)
CatalogRow = Tuple[str, str, str, float, float]


def _sort_key(row: CatalogRow):
    """Total order over candidates.

    ``place_id`` alone is not total — the source data may carry blank ids — so
    coordinates and name break ties. Without a total order the per-category
    ``random.sample`` below would depend on database row order.
    """
    place_id, name, category, lat, lon = row
    return (place_id or "", round(lat, 7), round(lon, 7), name or "", category)


def select_catalog_rows(net, rows: Optional[Sequence[dict]]) -> List[CatalogRow]:
    """Pick the catalog POIs for ``net`` from raw datasource ``rows``.

    Filters to the network's bounds, to categories the recommenders understand,
    and (on two-layer metros) to the principal-city core. Deduplicates by
    ``place_id``, then caps each category at ``POI_PARAMS["max_per_category"]``
    with a fixed-seed sample. Returns rows sorted by :func:`_sort_key`.
    """
    if not rows:
        return []

    south, west, north, east = net.bounds
    candidates: List[CatalogRow] = []
    seen = set()
    for row in rows:
        try:
            lat = float(row["latitude"])
            lon = float(row["longitude"])
        except (KeyError, ValueError, TypeError):
            continue
        if not (south <= lat <= north and west <= lon <= east):
            continue
        category = (row.get("category") or "").strip()
        if category not in params.CATEGORY_KEYWORDS:
            continue
        place_id = row.get("place_id", "") or ""
        if place_id and place_id in seen:
            continue
        if place_id:
            seen.add(place_id)
        candidates.append((place_id, row.get("name", "") or "", category, lat, lon))

    # Two-layer bound: POIs exist only in the principal city (no-op mask on
    # unlayered networks, where the bbox check above is the bound).
    if candidates and net.has_layers:
        keep = net.in_core_mask([c[3] for c in candidates], [c[4] for c in candidates])
        candidates = [c for c, k in zip(candidates, keep) if k]

    # Sort BEFORE grouping/sampling — this is what makes the cap reproducible.
    candidates.sort(key=_sort_key)

    by_category: Dict[str, List[CatalogRow]] = {}
    for cand in candidates:
        by_category.setdefault(cand[2], []).append(cand)

    cap = params.POI_PARAMS.get("max_per_category", 0)
    cap_rng = random.Random(CAP_SEED)
    selected: List[CatalogRow] = []
    for category in sorted(by_category):  # sorted: fixes node-id assignment order
        group = by_category[category]
        if cap and len(group) > cap:
            group = cap_rng.sample(group, cap)
        selected.extend(group)

    selected.sort(key=_sort_key)
    return selected


def catalog_fingerprint(selected: Sequence[CatalogRow]) -> str:
    """Stable hash of a selection, stored beside the warm graph and the routing
    matrices so a mismatched cache is detected rather than silently misread."""
    h = hashlib.sha256()
    h.update(str(params.POI_PARAMS.get("max_per_category", 0)).encode())
    for place_id, _name, category, lat, lon in selected:
        h.update(f"{place_id}|{category}|{lat:.7f}|{lon:.7f}\n".encode())
    return h.hexdigest()[:32]

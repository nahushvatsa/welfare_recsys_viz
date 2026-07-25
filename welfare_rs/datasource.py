"""Data access layer for metro geography, POIs, and personas.

Everything the simulation knows about the outside world — which counties feed
commuters into a principal city, where the leisure POIs are, and the persona
population — flows through one small interface so the backing store can change
without touching the model:

* :class:`LocalDataSource` (default) reads files on disk. None of them are in
  git — the county registry (``data/metro/metro_counties.json``) and the
  per-metro POI CSVs (``<cache>/pois/<metro>_leisure_pois.csv``) are local
  artifacts, regenerated or copied per machine.
* :class:`PostgresDataSource` reads the same shapes from the lab server's
  Postgres (ACS + LODES + POI tables) once it exists. Select it with
  ``WELFARE_RS_DATASOURCE=postgres`` and ``WELFARE_RS_PG_DSN=<dsn>``.

Expected Postgres schema (mirrors the local file schemas)::

    CREATE TABLE metro_county_flows (
        metro       text NOT NULL,   -- key from params.METRO_PARAMS["metros"]
        county_fips text NOT NULL,   -- 5-digit county / county-equivalent FIPS
        county_name text NOT NULL,
        geocode     text NOT NULL,   -- Nominatim query for the county polygon
        workers     integer NOT NULL,-- LODES: residents commuting into the core
        core_share  real NOT NULL DEFAULT 0,  -- share living inside the core
        PRIMARY KEY (metro, county_fips)
    );

    CREATE TABLE metro_pois (
        metro     text NOT NULL,
        place_id  text NOT NULL,
        name      text NOT NULL,
        category  text NOT NULL,     -- leisure category (params.CATEGORY_KEYWORDS)
        latitude  double precision NOT NULL,
        longitude double precision NOT NULL,
        PRIMARY KEY (metro, place_id)
    );

``metro_county_flows`` is the materialised form of the LODES query "workers by
home county whose work block falls inside the core city polygon"; the baked
JSON carries provisional hand estimates until that query can run for real.
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

from . import params


@dataclass(frozen=True)
class CountyFlow:
    """One county's inbound-commuter flow into a metro's core city."""

    fips: str
    name: str
    geocode: str
    workers: int
    core_share: float  # share of these workers whose home is inside the core


def select_counties(flows: List[CountyFlow], coverage: Optional[float] = None) -> List[CountyFlow]:
    """Apply the coverage rule: the smallest prefix of counties (sorted by
    inbound workers, descending) whose cumulative share reaches ``coverage``
    (default ``params.METRO_PARAMS["county_coverage"]``, i.e. 90%)."""
    if coverage is None:
        coverage = params.METRO_PARAMS["county_coverage"]
    ranked = sorted(flows, key=lambda f: f.workers, reverse=True)
    total = sum(f.workers for f in ranked)
    if total <= 0:
        return ranked
    kept, cum = [], 0
    for flow in ranked:
        kept.append(flow)
        cum += flow.workers
        if cum >= coverage * total:
            break
    return kept


class DataSource:
    """Interface. Implementations must be cheap to construct in worker
    processes (each seed worker builds its own from environment config)."""

    def county_flows(self, metro: str) -> List[CountyFlow]:
        """All known inbound flows for a metro (pre-coverage-cut)."""
        raise NotImplementedError

    def poi_rows(self, metro: str) -> Optional[List[dict]]:
        """Real leisure POIs for the metro's core city as dict rows with keys
        ``place_id, name, category, latitude, longitude`` — or None when the
        source has no POI data (simulation falls back to synthetic POIs)."""
        raise NotImplementedError

    def persona_csv_path(self) -> Optional[str]:
        """Behavioural persona CSV. Locations in it are ignored on metro
        networks (homes/works are sampled from the two-layer geography)."""
        raise NotImplementedError

    def has_pois(self, metro: str) -> bool:
        raise NotImplementedError


# ── Local disk (default) ─────────────────────────────────────────────────────

class LocalDataSource(DataSource):
    """Reads the gitignored on-disk artifacts."""

    def __init__(self, county_registry_path: Optional[str] = None,
                 cache_dir: Optional[str] = None):
        self._registry_path = county_registry_path or params.METRO_PARAMS["county_registry_path"]
        self._cache_dir = cache_dir or params.GEO_PARAMS["cache_dir"]
        self._registry: Optional[Dict[str, List[CountyFlow]]] = None

    def _load_registry(self) -> Dict[str, List[CountyFlow]]:
        if self._registry is None:
            if not os.path.exists(self._registry_path):
                raise FileNotFoundError(
                    f"County registry not found at {self._registry_path}. It is a "
                    "local (gitignored) data file — see welfare_rs/datasource.py "
                    "for its schema, or point WELFARE_RS_METRO_COUNTIES at it."
                )
            with open(self._registry_path, encoding="utf-8") as f:
                raw = json.load(f)
            self._registry = {
                metro: [
                    CountyFlow(
                        fips=str(c["fips"]), name=str(c["name"]),
                        geocode=str(c["geocode"]), workers=int(c["workers"]),
                        core_share=float(c.get("core_share", 0.0)),
                    )
                    for c in counties
                ]
                for metro, counties in raw.items() if not metro.startswith("_")
            }
        return self._registry

    def county_flows(self, metro: str) -> List[CountyFlow]:
        registry = self._load_registry()
        if metro not in registry:
            raise KeyError(f"No county flows for metro {metro!r} in {self._registry_path}")
        return registry[metro]

    def _poi_csv_candidates(self, metro: str) -> List[str]:
        """Candidate POI CSV paths, robust to a remapped HOME (matches the
        historic backend lookup). Legacy NYC filename accepted for 'nyc'."""
        names = [f"{metro}_leisure_pois.csv"]
        if metro == "nyc":
            names.append("nyc_leisure_pois.csv")
        dirs = [os.path.join(self._cache_dir, "pois")]
        try:
            import pwd

            real_home = pwd.getpwuid(os.getuid()).pw_dir
            dirs.append(os.path.join(real_home, ".cache", "welfare_rs", "pois"))
        except Exception:
            pass
        return [os.path.join(d, n) for d in dirs for n in names]

    def _poi_csv_path(self, metro: str) -> Optional[str]:
        for path in self._poi_csv_candidates(metro):
            if os.path.exists(path):
                return path
        return None

    def has_pois(self, metro: str) -> bool:
        return self._poi_csv_path(metro) is not None

    def poi_rows(self, metro: str) -> Optional[List[dict]]:
        path = self._poi_csv_path(metro)
        if path is None:
            return None
        with open(path, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def persona_csv_path(self) -> Optional[str]:
        path = params.NYC_PERSONA_CSV_PATH
        return path if os.path.exists(path) else None


# ── Postgres (the lab server) ────────────────────────────────────────────────

class PostgresDataSource(DataSource):
    """Same interface served from Postgres (schema in the module docstring).

    Connections are opened per query and closed immediately: queries are rare
    (once per worker per run) and short, and worker processes must not share
    sockets. Requires ``psycopg`` (v3) and ``WELFARE_RS_PG_DSN``.
    """

    def __init__(self, dsn: Optional[str] = None):
        self._dsn = dsn or os.environ.get("WELFARE_RS_PG_DSN")
        if not self._dsn:
            raise ValueError(
                "PostgresDataSource needs a DSN (WELFARE_RS_PG_DSN, e.g. "
                "'postgresql://user@host:5432/welfare')."
            )

    def _query(self, sql: str, args: tuple) -> List[tuple]:
        import psycopg

        with psycopg.connect(self._dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, args)
                return cur.fetchall()

    def county_flows(self, metro: str) -> List[CountyFlow]:
        rows = self._query(
            "SELECT county_fips, county_name, geocode, workers, core_share "
            "FROM metro_county_flows WHERE metro = %s ORDER BY workers DESC",
            (metro,),
        )
        if not rows:
            raise KeyError(f"No county flows for metro {metro!r} in Postgres")
        return [
            CountyFlow(fips=r[0], name=r[1], geocode=r[2],
                       workers=int(r[3]), core_share=float(r[4]))
            for r in rows
        ]

    def poi_rows(self, metro: str) -> Optional[List[dict]]:
        rows = self._query(
            "SELECT place_id, name, category, latitude, longitude "
            "FROM metro_pois WHERE metro = %s",
            (metro,),
        )
        if not rows:
            return None
        return [
            {"place_id": r[0], "name": r[1], "category": r[2],
             "latitude": r[3], "longitude": r[4]}
            for r in rows
        ]

    def has_pois(self, metro: str) -> bool:
        rows = self._query("SELECT 1 FROM metro_pois WHERE metro = %s LIMIT 1", (metro,))
        return bool(rows)

    def persona_csv_path(self) -> Optional[str]:
        # Personas stay file-based until the ACS-driven population lands.
        path = params.NYC_PERSONA_CSV_PATH
        return path if os.path.exists(path) else None


# ── Selection ────────────────────────────────────────────────────────────────

_DATASOURCE: Optional[DataSource] = None


def get_datasource() -> DataSource:
    """Environment-selected process-wide datasource.

    ``WELFARE_RS_DATASOURCE=postgres`` (with ``WELFARE_RS_PG_DSN``) switches to
    the server DB; anything else — including unset — means local disk.
    """
    global _DATASOURCE
    if _DATASOURCE is None:
        kind = os.environ.get("WELFARE_RS_DATASOURCE", "local").strip().lower()
        _DATASOURCE = PostgresDataSource() if kind == "postgres" else LocalDataSource()
    return _DATASOURCE

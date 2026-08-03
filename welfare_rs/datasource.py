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

    CREATE TABLE metro_od_pairs (
        metro    text     NOT NULL,
        h_geoid  char(15) NOT NULL,  -- 2020 Census block of the home
        w_geoid  char(15) NOT NULL,  -- 2020 Census block of the workplace
        h_county char(5)  NOT NULL,  -- left(h_geoid,5); filters to graph counties
        jobs     integer  NOT NULL,  -- LODES S000 on this pair = the draw weight
        sa01..sa03 integer,          -- worker age      <=29 / 30-54 / 55+
        se01..se03 integer,          -- monthly earnings <=1250 / 1251-3333 / >3333
        h_lat, h_lon, w_lat, w_lon double precision NOT NULL,
        PRIMARY KEY (metro, h_geoid, w_geoid)
    );

Both commute tables are derived from ONE filtered rowset — the LODES8 OD rows
whose *work* block falls inside the core polygon — at two levels of
aggregation. ``metro_county_flows`` is that set grouped by home county, and it
must exist first because the county list decides which counties are downloaded
into the road graph; a home outside the graph has no node to snap to.
``metro_od_pairs`` keeps the individual pairs and is what actually places
agents. See ``db/07_commute_tables.sql``.

A row of ``metro_od_pairs`` is NOT an agent: it says "this many jobs exist on
this home->work pair". A run draws ``num_agents`` samples from it weighted by
``jobs``, so population size stays a run parameter, independent of table size.
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from . import params


@dataclass(frozen=True)
class CountyFlow:
    """One county's inbound-commuter flow into a metro's core city."""

    fips: str
    name: str
    geocode: str
    workers: int
    core_share: float  # share of these workers whose home is inside the core


@dataclass(frozen=True)
class CommutePairs:
    """Block-level home->work commute pairs for one metro, as parallel arrays.

    Columnar rather than row-dicts because a metro can carry millions of pairs
    and only ``num_agents`` of them are ever drawn: arrays keep the memory flat
    and make the weighted draw a ``searchsorted`` over a prefix sum.

    ``jobs`` is the draw weight (LODES S000). ``age`` and ``earn`` are (n, 3)
    segment counts carried for a later ACS build, which will condition agent
    income and age on the commute instead of drawing them independently.
    """

    h_lat: np.ndarray
    h_lon: np.ndarray
    w_lat: np.ndarray
    w_lon: np.ndarray
    jobs: np.ndarray
    age: np.ndarray   # (n, 3): <=29 / 30-54 / 55+
    earn: np.ndarray  # (n, 3): <=$1250 / $1251-3333 / >$3333

    def __len__(self) -> int:
        return int(self.jobs.shape[0])


def _commute_pairs_from_rows(rows: Sequence[Sequence]) -> Optional[CommutePairs]:
    """Build :class:`CommutePairs` from ``(h_lat,h_lon,w_lat,w_lon,jobs,
    sa01..sa03,se01..se03)`` tuples. Returns None for an empty input."""
    if not rows:
        return None
    arr = np.asarray(rows, dtype=np.float64)
    return CommutePairs(
        h_lat=arr[:, 0].copy(),
        h_lon=arr[:, 1].copy(),
        w_lat=arr[:, 2].copy(),
        w_lon=arr[:, 3].copy(),
        jobs=arr[:, 4].astype(np.int64),
        age=arr[:, 5:8].astype(np.int32),
        earn=arr[:, 8:11].astype(np.int32),
    )


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

    def commute_pairs(self, metro: str,
                      counties: Optional[Sequence[str]] = None) -> Optional[CommutePairs]:
        """Real block-level home->work pairs for a metro, or None when the
        source has none (callers fall back to county-weighted home sampling
        plus a uniform workplace in the core).

        ``counties`` restricts homes to those 5-digit county FIPS — pass the
        counties actually built into the road graph, because a home outside it
        would snap to whatever node sits on the graph boundary.
        """
        return None


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

    def _od_csv_path(self, metro: str) -> Optional[str]:
        path = os.path.join(self._cache_dir, "od", f"{metro}_od_pairs.csv")
        return path if os.path.exists(path) else None

    def commute_pairs(self, metro: str,
                      counties: Optional[Sequence[str]] = None) -> Optional[CommutePairs]:
        """Local mirror of ``metro_od_pairs`` — same columns, one CSV per metro
        under ``<cache>/od/``. Absent on most machines, which is fine: the
        caller falls back to the county-weighted sampler."""
        path = self._od_csv_path(metro)
        if path is None:
            return None
        keep = set(counties) if counties else None
        rows = []
        with open(path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if keep is not None and r["h_county"] not in keep:
                    continue
                rows.append((
                    float(r["h_lat"]), float(r["h_lon"]),
                    float(r["w_lat"]), float(r["w_lon"]), int(r["jobs"]),
                    int(r["sa01"]), int(r["sa02"]), int(r["sa03"]),
                    int(r["se01"]), int(r["se02"]), int(r["se03"]),
                ))
        return _commute_pairs_from_rows(rows)


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
        # ORDER BY is load-bearing, not cosmetic: the per-category cap in
        # welfare_rs.poi_select samples BY POSITION, and the POI set fixes the
        # graph's node ids, which index the precomputed routing matrices. An
        # unordered scan is stable in practice but promises nothing — one table
        # rewrite would silently select a different catalog.
        rows = self._query(
            "SELECT place_id, name, category, latitude, longitude "
            "FROM metro_pois WHERE metro = %s ORDER BY place_id, latitude, longitude",
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

    # Streamed in batches this size, into arrays preallocated from COUNT(*).
    _OD_BATCH = 100_000

    def commute_pairs(self, metro: str,
                      counties: Optional[Sequence[str]] = None) -> Optional[CommutePairs]:
        """Stream the metro's pairs into preallocated arrays.

        A big metro carries millions of pairs and every seed worker loads its
        own copy inside one 24 GiB service cgroup, so this deliberately avoids
        ``fetchall()``: materialising millions of Python tuples costs several
        times what the finished arrays do. A server-side (named) cursor keeps
        peak memory at ``final arrays + one batch``.
        """
        import psycopg

        cols = ("h_lat, h_lon, w_lat, w_lon, jobs, "
                "sa01, sa02, sa03, se01, se02, se03")
        where = "metro = %s"
        args: tuple = (metro,)
        if counties:
            where += " AND h_county = ANY(%s)"
            args = (metro, list(counties))

        with psycopg.connect(self._dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(f"SELECT count(*) FROM metro_od_pairs WHERE {where}", args)
                n = int(cur.fetchone()[0])
            if n == 0:
                return None

            h_lat = np.empty(n, dtype=np.float64)
            h_lon = np.empty(n, dtype=np.float64)
            w_lat = np.empty(n, dtype=np.float64)
            w_lon = np.empty(n, dtype=np.float64)
            jobs = np.empty(n, dtype=np.int64)
            age = np.empty((n, 3), dtype=np.int32)
            earn = np.empty((n, 3), dtype=np.int32)

            i = 0
            # Named cursor => the server holds the result set and ships batches.
            with conn.cursor(name=f"od_{metro}") as cur:
                cur.itersize = self._OD_BATCH
                cur.execute(f"SELECT {cols} FROM metro_od_pairs WHERE {where}", args)
                while True:
                    batch = cur.fetchmany(self._OD_BATCH)
                    if not batch:
                        break
                    a = np.asarray(batch, dtype=np.float64)
                    k = a.shape[0]
                    if i + k > n:  # table grew mid-read; keep what fits
                        k = n - i
                        a = a[:k]
                        if k == 0:
                            break
                    h_lat[i:i + k] = a[:, 0]
                    h_lon[i:i + k] = a[:, 1]
                    w_lat[i:i + k] = a[:, 2]
                    w_lon[i:i + k] = a[:, 3]
                    jobs[i:i + k] = a[:, 4]
                    age[i:i + k] = a[:, 5:8]
                    earn[i:i + k] = a[:, 8:11]
                    i += k

        if i == 0:
            return None
        if i < n:  # rows vanished between the count and the read
            h_lat, h_lon = h_lat[:i], h_lon[:i]
            w_lat, w_lon = w_lat[:i], w_lon[:i]
            jobs, age, earn = jobs[:i], age[:i], earn[:i]
        return CommutePairs(h_lat=h_lat, h_lon=h_lon, w_lat=w_lat, w_lon=w_lon,
                            jobs=jobs, age=age, earn=earn)

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

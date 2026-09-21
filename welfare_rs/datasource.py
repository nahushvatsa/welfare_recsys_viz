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
  Postgres (survey + LODES + POI tables). Select it with
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

The population is served by a VIEW rather than a table::

    survey.personas   -- 473 rows, one per survey respondent, 53 columns

It is the SQL twin of ``data/build_survey_personas.py``, defined in
``db/10_survey_personas.sql`` over the respondent-level ``survey.respondents``
loaded by ``db/load_survey.py``. Three things about it are deliberate:

* **A view, not a table.** The population is the survey sample itself, not a
  synthesised draw, so there is nothing to materialise; 473 rows recomputed on
  demand cost nothing and cannot drift from the respondent file.
* **It is the access boundary.** ``survey.respondents`` is human-subjects data
  under IRB-FY2026-11354 and the API-facing ``welfare_app`` role cannot read
  it. A Postgres view executes with its OWNER's privileges, so ``welfare_app``
  selects these 53 columns while holding no grant on the 136-column file
  underneath. This is checked, not assumed — see db/09_survey_schema.sql.
* **Row order is part of the contract**, because agent ``i`` is respondent
  ``i``. See :meth:`DataSource.personas`.

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
import warnings
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


# Column -> parser for a population row. Applied by BOTH datasources, which is
# what makes a CSV-backed population and a Postgres-backed one indistinguishable
# to the engine. A column absent from this map is passed through as text.
_POPULATION_TYPES = {
    "replicate": int, "agent_id": int, "is_worker": bool,
    "home_lat": float, "home_lon": float,
    "work_lat": float, "work_lon": float,
    "sporder": int, "age": int, "sex": int, "employed": bool,
    "industry_seg": int, "own_earnings": float, "hh_income": float,
    "vehicles": int, "disability": bool, "ambulatory": bool, "hh_size": int,
    "commute_mode": int, "commute_min_implied": float,
    "commute_min_reported": float, "transit_share": float,
    "match_level": int,
}

# A column missing from the map above is passed through as TEXT, and a boolean
# is the dangerous case: the string "False" is truthy, so an omitted
# `ambulatory` silently gave every agent wheelchair mobility needs. Caught end
# to end rather than by inspection, which is why
# scripts/verify_acs_population.py now asserts that every non-text column of
# public.metro_population appears here.

_TRUE_TOKENS = {"t", "true", "1", "yes", "y"}


def coerce_population_row(row: dict) -> dict:
    """Type one population row, from either a CSV cell or a DB value.

    Empty strings become None, because a CSV cannot distinguish "" from NULL
    and the engine must see the same absence either way (a non-worker has no
    workplace, and no vehicles recorded if its household is group quarters).
    """
    out = {}
    for key, value in row.items():
        if value is None or (isinstance(value, str) and value.strip() == ""):
            out[key] = None
            continue
        parser = _POPULATION_TYPES.get(key)
        if parser is None:
            out[key] = value if isinstance(value, str) else str(value)
        elif parser is bool:
            out[key] = value if isinstance(value, bool) else str(value).strip().lower() in _TRUE_TOKENS
        else:
            out[key] = parser(value)
    return out


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

    def personas(self) -> Optional[List[dict]]:
        """The agent population as dict rows, or None when the source has none.

        One row per survey respondent, in the column layout
        ``Simulation._load_personas`` consumes. Prefer this over
        :meth:`persona_csv_path` — it is the only form the Postgres source can
        serve, since there the population is a view rather than a file.

        ROW ORDER IS LOAD-BEARING, not cosmetic. ``Simulation`` assigns
        ``personas[i]`` to agent ``i`` for the first ``len(personas)`` agents,
        and the paired No-RS counterfactual compares agent ``i`` across
        conditions — so agent ``i`` must be the same respondent in every run, or
        the pairing compares different people. Implementations must return a
        deterministic order (respondent id ascending).

        VALUES ARE STRINGS, exactly as ``csv.DictReader`` yields them, with the
        empty string for a missing cell. That is a compatibility contract, not
        laziness: it makes the DB path and the CSV path produce identical dicts,
        so no consumer has to care which one it got. Python's ``str`` of a float
        is shortest-round-trip, so ``float(str(x)) == x`` bit for bit.
        """
        return None

    def has_pois(self, metro: str) -> bool:
        raise NotImplementedError

    def population(self, metro: str, replicate: int = 0,
                   limit: Optional[int] = None) -> Optional[List[dict]]:
        """A pre-built agent population for a metro, or None when absent.

        One row per agent, each carrying its geography (LODES home and work, or
        a core home block for a non-worker), the ACS/PUMS person it copies, and
        the survey respondent supplying its psychometrics. Built offline by
        ``db/build_population.py``; see the schema in db/14_population_schema.sql.

        Nothing is sampled at run time. The draw needs reweighted PUMS, ACS
        controls and the whole LODES pair table — a run has no business holding
        any of that, and repeating the work per seed and per condition would be
        both slow and a source of drift between conditions that must stay
        comparable.

        ROW ORDER IS LOAD-BEARING, exactly as in :meth:`personas`: agent ``i``
        is the row with ``agent_id = i``, and the paired No-RS counterfactual
        compares agent ``i`` across conditions. ``limit`` takes a prefix, so a
        smaller run is a strict subset of a larger one rather than a different
        population.

        ``replicate`` is what a simulation seed used to do to the population:
        seed ``s`` reads replicate ``s``, so seeds still vary who the agents are
        as well as how they behave.

        VALUES ARE NATIVELY TYPED here — ints, floats, bools, None — unlike
        :meth:`personas`, which must mimic ``csv.DictReader`` strings for
        compatibility with the file path it replaced. Both implementations run
        their rows through :func:`coerce_population_row`, so the two paths
        produce identical dicts by construction rather than by agreement.
        """
        return None

    def population_replicates(self, metro: str) -> int:
        """How many built population replicates a metro has, 0 if none.

        A run picks one by seed, so it needs the count to map an arbitrary seed
        onto a replicate that exists.
        """
        return 0

    def home_blocks(self, metro: str) -> Optional[List[tuple]]:
        """Populated CORE blocks as ``(lat, lon)``, or None when absent.

        Needed by the routing precompute, not by the model: every point an
        agent home can snap to must be inside the precomputed endpoint universe
        (welfare_rs/routing_matrix.py), or those agents fall through to the
        lazy per-source Dijkstra, which is quadratic in the agent count. Worker
        homes come from the commute pairs and are already covered; non-worker
        homes are drawn from these blocks and are not.
        """
        return None

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
        # Survey-grounded population (one profile per respondent). Falls back to
        # the synthetic file when it has not been built — see
        # data/build_survey_personas.py.
        for path in (params.SURVEY_PERSONA_CSV_PATH, params.NYC_PERSONA_CSV_PATH):
            if os.path.exists(path):
                return path
        return None

    def personas(self) -> Optional[List[dict]]:
        """Read the persona CSV. File order is respondent order (R001..R473)."""
        path = self.persona_csv_path()
        if path is None:
            return None
        with open(path, newline="", encoding="utf-8") as f:
            return [r for r in csv.DictReader(f) if r.get("PersonaID")]

    def population(self, metro: str, replicate: int = 0,
                   limit: Optional[int] = None) -> Optional[List[dict]]:
        """Read a population exported to CSV by ``db/export_population.py``.

        This is the no-database path: the file is a dump of one
        ``(metro, replicate)`` slice of public.metro_population, already in
        agent_id order. The order is re-asserted here anyway, because a file on
        disk can be edited and the contract cannot rely on how it was written.
        """
        path = os.path.join(params.POPULATION_DIR,
                            f"{metro}_r{int(replicate):02d}.csv")
        if not os.path.exists(path):
            return None
        with open(path, newline="", encoding="utf-8") as f:
            rows = [coerce_population_row(r) for r in csv.DictReader(f)]
        rows.sort(key=lambda r: r["agent_id"])
        if limit is not None:
            rows = rows[:int(limit)]
        return rows or None

    def population_replicates(self, metro: str) -> int:
        import glob

        return len(glob.glob(os.path.join(params.POPULATION_DIR, f"{metro}_r*.csv")))

    def home_blocks(self, metro: str) -> Optional[List[tuple]]:
        path = os.path.join(params.POPULATION_DIR, f"{metro}_home_blocks.csv")
        if not os.path.exists(path):
            return None
        with open(path, newline="", encoding="utf-8") as f:
            return [(float(r["lat"]), float(r["lon"])) for r in csv.DictReader(f)] or None

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

    def population(self, metro: str, replicate: int = 0,
                   limit: Optional[int] = None) -> Optional[List[dict]]:
        """Read a built population from ``public.metro_population``.

        The ORDER BY and the ``agent_id`` bound are the row-order contract in
        :meth:`DataSource.population`; ``LIMIT`` alone would not do, because it
        is only a prefix of *some* order unless the order is stated here.

        Returns None when the metro has no population built, which is a real
        state and not an error: the caller then falls back to the survey
        persona population plus run-time commute sampling.
        """
        import psycopg

        sql = ("SELECT * FROM public.metro_population "
               "WHERE metro = %s AND replicate = %s ORDER BY agent_id")
        args = (metro, int(replicate))
        if limit is not None:
            sql += " LIMIT %s"
            args = args + (int(limit),)
        try:
            rows = self._query_named(sql, args)
        except (psycopg.errors.UndefinedTable,
                psycopg.errors.InsufficientPrivilege) as exc:
            warnings.warn(
                f"public.metro_population is unreadable ({exc.__class__.__name__}); "
                f"falling back to the survey persona population for {metro}.",
                RuntimeWarning, stacklevel=2,
            )
            return None
        return [coerce_population_row(r) for r in rows] or None

    def population_replicates(self, metro: str) -> int:
        import psycopg

        try:
            rows = self._query("SELECT count(DISTINCT replicate) "
                               "FROM public.metro_population WHERE metro = %s", (metro,))
        except (psycopg.errors.UndefinedTable, psycopg.errors.InsufficientPrivilege):
            return 0
        return int(rows[0][0]) if rows else 0

    def home_blocks(self, metro: str) -> Optional[List[tuple]]:
        rows = self._query(
            "SELECT lat, lon FROM public.metro_home_blocks WHERE metro = %s "
            "ORDER BY geoid", (metro,))
        return [(float(lat), float(lon)) for lat, lon in rows] or None

    def persona_csv_path(self) -> Optional[str]:
        # Kept for callers that genuinely want a path (scripts/bench_routing.py,
        # scripts/verify_routing_matrix.py). The DB population is served by
        # personas() below; this is the on-disk fallback and may be absent on a
        # server that only ever reads Postgres.
        for path in (params.SURVEY_PERSONA_CSV_PATH, params.NYC_PERSONA_CSV_PATH):
            if os.path.exists(path):
                return path
        return None

    def personas(self) -> Optional[List[dict]]:
        """Read the survey population from ``survey.personas``.

        That view is the SQL twin of ``data/build_survey_personas.py``;
        ``scripts/verify_survey_personas_view.py`` asserts the two agree on all
        25,069 cells, so this path and the CSV path build the same agents.

        ``ORDER BY "PersonaID"`` is repeated here even though the view already
        carries one: a view's ORDER BY is not contractual through an outer
        query, and this order decides which respondent becomes agent i (see
        :meth:`DataSource.personas`).

        Falls back to the CSV only when the view is genuinely not there — a
        server that has the POI and commute tables but not yet the survey
        schema. The catch is NARROW and LOUD on purpose: an outage or a
        credentials problem must propagate, not quietly swap in whatever CSV
        happens to be on that disk. A silent fallback is how a run stops being
        the run you think it is.
        """
        import psycopg

        try:
            rows = self._query_named(
                'SELECT * FROM survey.personas ORDER BY "PersonaID"', ())
        except (psycopg.errors.UndefinedTable,        # 42P01 no such view
                psycopg.errors.InvalidSchemaName,     # 3F000 no survey schema
                psycopg.errors.InsufficientPrivilege  # 42501 not granted
                ) as exc:
            path = self.persona_csv_path()
            warnings.warn(
                f"survey.personas is unreadable ({exc.__class__.__name__}); "
                f"falling back to {path or 'no persona file'}. The population "
                f"is NOT coming from the database.",
                RuntimeWarning, stacklevel=2,
            )
            if path is None:
                return None
            with open(path, newline="", encoding="utf-8") as f:
                return [r for r in csv.DictReader(f) if r.get("PersonaID")]
        # NULL -> '' and every value to str: see the contract in
        # DataSource.personas. csv.DictReader yields strings, and matching it
        # exactly is what lets the two sources be interchangeable.
        return [{k: ("" if v is None else str(v)) for k, v in row.items()}
                for row in rows]

    def _query_named(self, sql: str, args: tuple) -> List[dict]:
        """Like _query but returns dict rows keyed by the result column names."""
        import psycopg

        with psycopg.connect(self._dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, args)
                cols = [d.name for d in cur.description]
                return [dict(zip(cols, r)) for r in cur.fetchall()]


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

"""Prefetch and warm-cache the two-layer road networks for all metros.

For each metro this geocodes the boundary polygons (Nominatim), downloads the
core drive network + arterial shell (Overpass), composes/simplifies them, bakes
the catalog POIs into the graph, and writes the disk caches (boundaries /
GraphML / warm pickle) so runs from the app are purely local afterwards. It
does the same for the core's walk and bike networks (welfare_rs.active_modes).

Pin the Overpass mirror with $WELFARE_RS_OVERPASS_URL for bulk warming; the
probe can otherwise pick overpass-api.de, which blocks this host after a burst
of queries.

It then precomputes the **routing matrices** — all-pairs shortest distance and
fastest time over the metro's endpoint universe, plus shortest walk and bike
distance over its core endpoints (see ``welfare_rs.routing_matrix``). That is the expensive, reusable part: once it
exists, a simulation never runs Dijkstra again, so concurrent runs and large
agent populations cost nothing extra.

First run of everything is network-bound and can take an hour or more — run it
once, ideally overnight:

    python scripts/warm_metros.py                    # all metros
    python scripts/warm_metros.py miami seattle      # a subset
    python scripts/warm_metros.py --graph-only       # skip the matrices
    python scripts/warm_metros.py --rebuild-matrices # force recomputation
    python scripts/warm_metros.py --workers 16       # cap the fan-out
"""

from __future__ import annotations

import os
import sys
import time

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)


def _load_env() -> None:
    """Load the server datasource credentials before anything imports them.

    Without this the datasource silently falls back to LocalDataSource, which
    serves no commute pairs and (for most metros) no POIs — and a rebuild would
    then overwrite good warm caches with empty ones. scripts/warm_all.sh sources
    these files itself; doing it here too means running this module directly is
    equally safe.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    for name in (".env.server", ".env"):
        path = os.path.join(_REPO, name)
        if os.path.exists(path):
            load_dotenv(path, override=False)
    if os.environ.get("WELFARE_RS_PG_DSN") and not os.environ.get("WELFARE_RS_DATASOURCE"):
        os.environ["WELFARE_RS_DATASOURCE"] = "postgres"


_load_env()

from welfare_rs import params, routing_matrix  # noqa: E402
from welfare_rs.datasource import get_datasource  # noqa: E402
from welfare_rs.metro import build_metro_network, ensure_routing_matrices  # noqa: E402


def _preflight(metros, allow_degraded: bool) -> None:
    """Fail before touching any cache if the datasource cannot serve a metro.

    Cheap checks only — county flows and POI presence. The expensive commute
    table is validated later by build_metro_network, which refuses to write a
    degraded pickle.
    """
    ds = get_datasource()
    print(f"datasource: {type(ds).__name__}", flush=True)
    if allow_degraded:
        return
    problems = []
    for metro in metros:
        try:
            if not ds.county_flows(metro):
                problems.append(f"{metro}: no county flows")
        except Exception as exc:
            problems.append(f"{metro}: county flows unavailable ({exc})")
        try:
            if not ds.has_pois(metro):
                problems.append(f"{metro}: no POI rows")
        except Exception as exc:
            problems.append(f"{metro}: POI lookup failed ({exc})")
    if problems:
        print("\nDatasource cannot serve these metros:", file=sys.stderr)
        for p in problems:
            print(f"  {p}", file=sys.stderr)
        print(
            "\nFor the server data set export WELFARE_RS_DATASOURCE=postgres and "
            "WELFARE_RS_PG_DSN (see .env.server), or pass --allow-degraded to "
            "build from whatever is available.",
            file=sys.stderr,
        )
        sys.exit(2)


def main() -> None:
    args = sys.argv[1:]
    graph_only = "--graph-only" in args
    rebuild = "--rebuild-matrices" in args
    allow_degraded = "--allow-degraded" in args
    workers = None
    if "--workers" in args:
        i = args.index("--workers")
        workers = int(args[i + 1])
        del args[i:i + 2]
    metros = [a for a in args if not a.startswith("--")] or list(params.METRO_PARAMS["metros"])

    _preflight(metros, allow_degraded)

    failures = []
    for metro in metros:
        label = params.METRO_PARAMS["metros"][metro]["label"]
        print(f"\n=== {metro} ({label}) ===", flush=True)
        t0 = time.time()
        try:
            net = build_metro_network(metro)
        except Exception as exc:  # keep going; report at the end
            failures.append((metro, f"{type(exc).__name__}: {exc}"))
            print(f"  FAILED: {exc}", flush=True)
            continue
        core = len(net._core_nodes or [])
        commutes = (f" · {net.num_commute_pairs:,} commute pairs"
                    if net.has_commutes else " · NO commute pairs (fallback sampling)")
        print(
            f"  {net.num_base_nodes:,} intersections ({core:,} in core) · "
            f"{net.num_edges:,} edges · {len(net._poi_nodes):,} POI nodes"
            f"{commutes} · {time.time() - t0:.0f}s",
            flush=True,
        )
        for mode, mnet in sorted(net.mode_networks.items()):
            print(f"  {mode}: {mnet.num_base_nodes:,} nodes · {mnet.num_edges:,} edges · "
                  f"{len(mnet._poi_nodes):,} POI nodes", flush=True)

        if graph_only:
            continue
        try:
            ensure_routing_matrices(metro, workers=workers, rebuild=rebuild)
        except Exception as exc:
            failures.append((metro, f"matrices: {type(exc).__name__}: {exc}"))
            print(f"  MATRIX FAILED: {exc}", flush=True)

    print("\n" + "=" * 60)
    for metro in metros:
        for mode in ("drive", *params.ACTIVE_NETWORK_PARAMS["modes"]):
            meta = routing_matrix.read_meta(params.GEO_PARAMS["cache_dir"], metro, mode)
            label = f"{metro}/{mode}"
            if meta:
                tables = 3 if meta.get("has_time") else 1
                gb = meta["universe"] ** 2 * 8 * tables / 1e9
                print(f"  {label:15} universe {meta['universe']:>7,}  {gb:6.1f} GB  "
                      f"built in {meta.get('build_seconds', 0):.0f}s")
            else:
                print(f"  {label:15} no routing matrices")

    if failures:
        print("\nFailed metros:")
        for metro, err in failures:
            print(f"  {metro}: {err}")
        sys.exit(1)
    print("\nAll metros warmed.")


if __name__ == "__main__":
    main()

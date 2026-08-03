"""Measure what the precomputed routing matrices are worth, end to end.

Runs the same simulation twice on the same warmed network: once served by the
matrices, once with them detached so routing falls back to per-source Dijkstra.
Both paths must produce identical metrics — a speedup that changes the answers
would not be a speedup.

    python scripts/bench_routing.py nyc [agents] [days]
"""

from __future__ import annotations

import os
import sys
import time

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)


def _load_env() -> None:
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

from welfare_rs import params  # noqa: E402
from welfare_rs.datasource import get_datasource  # noqa: E402
from welfare_rs.experiment_harness import table1_metrics  # noqa: E402
from welfare_rs.metro import build_metro_network  # noqa: E402
from welfare_rs.simulation import Simulation  # noqa: E402


def run_once(metro: str, agents: int, days: int, use_matrix: bool) -> tuple:
    ds = get_datasource()
    net = build_metro_network(metro, cache_dir=params.GEO_PARAMS["cache_dir"])
    if not use_matrix:
        net._mat_nodes = net._mat_dist_m = net._mat_car_sec = net._mat_car_m = None
        net._mat_rows = {}

    t0 = time.time()
    sim = Simulation(
        num_agents=agents, seed=42, use_recommenders=True,
        persona_csv_path=ds.persona_csv_path(), road_network=net,
        poi_rows=ds.poi_rows(metro), disabled_modes=("transit",),
    )
    init_s = time.time() - t0

    t0 = time.time()
    sim.run_days(days)
    run_s = time.time() - t0

    return init_s, run_s, table1_metrics(sim), len(net._ss_cache), len(net._ss_time_cache)


def main() -> None:
    metro = sys.argv[1] if len(sys.argv) > 1 else "miami"
    agents = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    days = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    params.SIMPLIFICATION_TOGGLES["CAR_ONLY_MODE"] = True

    print(f"{metro}: {agents} agents x {days} days\n")

    i_m, r_m, m_m, ss_m, sst_m = run_once(metro, agents, days, use_matrix=True)
    print(f"  matrix : init {i_m:6.1f}s  run {r_m:7.1f}s  total {i_m + r_m:7.1f}s"
          f"   live dijkstras: {ss_m + sst_m}")

    i_l, r_l, m_l, ss_l, sst_l = run_once(metro, agents, days, use_matrix=False)
    print(f"  lazy   : init {i_l:6.1f}s  run {r_l:7.1f}s  total {i_l + r_l:7.1f}s"
          f"   live dijkstras: {ss_l + sst_l}")

    speedup = (i_l + r_l) / max(i_m + r_m, 1e-9)
    print(f"\n  speedup: {speedup:.1f}x")

    same = m_m == m_l
    print(f"  metrics identical: {same}")
    if not same:
        print(f"    matrix: {m_m}")
        print(f"    lazy  : {m_l}")
        sys.exit(1)


if __name__ == "__main__":
    main()

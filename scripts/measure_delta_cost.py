"""Measure the spread of dC, the extra travel cost of a recommendation, to set
``SIMPLIFIED_ETA_PARAMS["beta_delta_cost"]``.

``beta_delta_cost`` turns dC (the planning price of the offered venue minus
that of the agent's own pick) into a change in the probability of accepting.
The paper gives no value and no study estimates one, so it is SPECIFIED
against a design target: the gap in mean dC between a proximity-led ranker
(offers close by) and a footfall-led one (offers what is popular, wherever it
is) should become a ~0.195 gap in eta, i.e. a ~20-point acceptance difference.
That is large enough for distance to matter and stays below the trust term's
+/-0.20, so it informs adoption without dominating it. The value was first set
at 0.30 when dC was priced at 0.12 per drive km (gap 0.650 in Manhattan). The
planning price has since changed to expected generalised cost across modes,
which changes dC's scale, so the value is re-derived here the same way.

Every offer is recorded where the agent prices it: dC is the argument
``Agent._estimate_eta`` receives. Offers the agent cannot reach at all are
declined before pricing and never reach it.

    python scripts/measure_delta_cost.py [metro] [agents] [days] [seed]
"""

from __future__ import annotations

import os
import sys

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)


def _load_env() -> None:
    """Same datasource bootstrap as scripts/warm_metros.py."""
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
from welfare_rs.agent import Agent  # noqa: E402
from welfare_rs.datasource import get_datasource  # noqa: E402
from welfare_rs.metro import build_metro_network  # noqa: E402
from welfare_rs.recommender_systems import RECOMMENDER_PRESETS, RecommenderConfig  # noqa: E402
from welfare_rs.simulation import Simulation  # noqa: E402

# The design target: 0.30 x 0.650, the eta gap the original value produced.
TARGET_ETA_GAP = 0.30 * 0.650
ARMS = ("Proximity-led", "Footfall-led")


def measure(metro: str, n_agents: int, n_days: int, seed: int) -> dict:
    ds = get_datasource()
    net = build_metro_network(metro)
    rows = ds.population(metro, replicate=seed % ds.population_replicates(metro),
                         limit=n_agents + max(100, n_agents // 5))
    recorded = []
    original = Agent._estimate_eta

    def recording(self, recommendation_score, delta_cost=0.0):
        eta = original(self, recommendation_score, delta_cost=delta_cost)
        recorded.append((delta_cost, eta))
        return eta

    out = {}
    Agent._estimate_eta = recording
    try:
        for arm in ARMS:
            recorded.clear()
            sim = Simulation(
                num_agents=n_agents, seed=seed, use_recommenders=True,
                persona_rows=ds.personas(), population_rows=rows, road_network=net,
                poi_rows=ds.poi_rows(metro),
                recommender_config=RecommenderConfig(label=arm, **RECOMMENDER_PRESETS[arm]),
            )
            sim.run_days(n_days)
            dc = np.array([r[0] for r in recorded])
            eta = np.array([r[1] for r in recorded])
            out[arm] = {"n": len(dc), "dc_mean": float(dc.mean()),
                        "dc_p10": float(np.percentile(dc, 10)),
                        "dc_p50": float(np.median(dc)),
                        "dc_p90": float(np.percentile(dc, 90)),
                        "eta_mean": float(eta.mean())}
    finally:
        Agent._estimate_eta = original
    return out


def main() -> None:
    metro = sys.argv[1] if len(sys.argv) > 1 else "nyc"
    n_agents = int(sys.argv[2]) if len(sys.argv) > 2 else 400
    n_days = int(sys.argv[3]) if len(sys.argv) > 3 else 6
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 42
    beta = params.SIMPLIFIED_ETA_PARAMS["beta_delta_cost"]
    res = measure(metro, n_agents, n_days, seed)
    print(f"{metro}: {n_agents} agents x {n_days} days, seed {seed}, "
          f"current beta_delta_cost {beta}")
    for arm, r in res.items():
        print(f"  {arm:14} offers {r['n']:5}  dC mean {r['dc_mean']:+.3f}  "
              f"p10 {r['dc_p10']:+.3f} p50 {r['dc_p50']:+.3f} p90 {r['dc_p90']:+.3f}  "
              f"mean eta {r['eta_mean']:.3f}")
    gap = res["Footfall-led"]["dc_mean"] - res["Proximity-led"]["dc_mean"]
    print(f"  gap in mean dC: {gap:.3f}")
    if gap > 0:
        print(f"  beta for a {TARGET_ETA_GAP:.3f} eta gap: {TARGET_ETA_GAP / gap:.3f}")


if __name__ == "__main__":
    main()

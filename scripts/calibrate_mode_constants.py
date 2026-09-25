"""Calibrate the walk and bike alternative-specific constants per metro.

Mode choice carries many hand-set mode terms (comfort, status, enjoyment,
community and survey biases, leisure taste shifts). Nothing ties their net
effect to how people in a given city actually travel, so each metro gets one
additive constant per active mode (car is the reference, 0), chosen so that
simulated COMMUTE walk and bike shares match observed ones. This is the
standard way a mode-choice model is fitted to a new region: iterate
``ASC_m += ln(observed_m / simulated_m)`` until the shares agree (Train 2009,
*Discrete Choice Methods with Simulation*, §2.8).

Only commuters who actually face a choice are fitted. A constant can only move
someone with two or more permitted modes; an agent with one (a carless agent
outside NYC whose commute is too far to walk has only bike-share) contributes a
fixed share whatever the constant is. Including them made the target
unreachable: in Miami 2.1% of core commuters are bike-only, most of whom really
ride transit, more than the 1.15% who actually bike, and the bike constant was
driven to its -15 bound — which would have made cycling all but unchoosable for
everyone who does have a choice, leisure trips included. Single-option
commuters are reported separately.

Observed shares come from the agents themselves. Every agent is a PUMS person
carrying its own means of transportation to work (JWTRNS), so the target is
measured on exactly the people being simulated: core-resident workers with a
workplace, excluding those who work from home or report "other". Transit
commuters stay in the denominator. With no transit mode yet, the constants
match walk and bike shares and "car" absorbs everyone else, transit riders
included. That is a stated limitation, not something calibration can fix.

Simulated shares are EXPECTED shares, not sampled ones. For each agent the
morning commute's options are built exactly as a departure builds them, and
the agent's own decision rule is turned into choice probabilities by
``Simulation.mode_choice_probabilities`` (the same code the planner prices
venues with). So the fit has no Monte Carlo noise. Road congestion is left at free flow,
which is what a departure sees at these population sizes (capacity scales
with the metro graph's tens of thousands of intersections).

    python scripts/calibrate_mode_constants.py                 # all metros
    python scripts/calibrate_mode_constants.py miami chicago --agents 8000
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time

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
from welfare_rs.datasource import get_datasource  # noqa: E402
from welfare_rs.metro import build_metro_network  # noqa: E402
from welfare_rs.simulation import Simulation  # noqa: E402

ACTIVE = ("walk", "bike")
JWTRNS_WALK, JWTRNS_BIKE = 10, 9
JWTRNS_EXCLUDED = (11, 12)  # worked from home, other


def commuters(sim):
    """(agent, home activity, work activity) for core-resident workers whose
    day-0 plan includes a real commute and whose PUMS mode is informative."""
    out = []
    for a in sim.agents:
        cm = a.characteristics.get("commute_mode")
        if not a.lives_in_core or cm is None or int(cm) in JWTRNS_EXCLUDED:
            continue
        s = a.schedule
        if len(s) < 2 or s[1].type != "work" or s[1].location == a.home:
            continue
        out.append((a, s[0], s[1]))
    return out


def with_choice(sim, group):
    """Split commuters into those with 2+ permitted modes and those with one.

    Choice sets depend on availability only, never on the constants, so the
    split is computed once.
    """
    choosing, single = [], []
    for agent, home, work in group:
        agent.car_out = False
        agent.current_activity_index = 0
        options = sim._departure_outcomes(agent, home, work)
        (choosing if len(options) > 1 else single).append((agent, home, work, next(iter(options))))
    return [(a, h, w) for a, h, w, _m in choosing], single


def observed_shares(group):
    n = len(group)
    walk = sum(1 for a, _h, _w in group if int(a.characteristics["commute_mode"]) == JWTRNS_WALK)
    bike = sum(1 for a, _h, _w in group if int(a.characteristics["commute_mode"]) == JWTRNS_BIKE)
    return {"walk": walk / n, "bike": bike / n}


def expected_shares(sim, group):
    totals = {}
    for agent, home, work in group:
        agent.car_out = False
        agent.current_activity_index = 0
        outcomes = sim._departure_outcomes(agent, home, work)
        for m, p in sim.mode_choice_probabilities(agent, outcomes).items():
            totals[m] = totals.get(m, 0.0) + p
    n = len(group)
    return {m: v / n for m, v in totals.items()}


def calibrate(metro, n_agents, replicates, max_iter=40, tol=0.001, log=print):
    ds = get_datasource()
    net = build_metro_network(metro)
    sims = []
    for rep in replicates:
        rows = ds.population(metro, replicate=rep, limit=n_agents + max(100, n_agents // 5))
        sims.append(Simulation(num_agents=n_agents, seed=1000 + rep, use_recommenders=False,
                               persona_rows=ds.personas(), population_rows=rows,
                               road_network=net, poi_rows=ds.poi_rows(metro)))
    all_commuters = [commuters(s) for s in sims]
    split = [with_choice(s, grp) for s, grp in zip(sims, all_commuters)]
    groups = [c for c, _single in split]
    singles = [x for _c, single in split for x in single]
    group_all = [g for grp in groups for g in grp]
    target = observed_shares(group_all)
    n = len(group_all)
    n_total = sum(len(g) for g in all_commuters)
    only = {}
    for _a, _h, _w, mode in singles:
        only[mode] = only.get(mode, 0) + 1
    log(f"{metro}: {n:,} of {n_total:,} core-resident commuters have a choice "
        f"(single-option: " + ", ".join(f"{m}-only {c / n_total:.1%}" for m, c in sorted(only.items()))
        + f"); dropped rows: {sum(s.dropped_population_rows for s in sims)} — observed "
        f"walk {target['walk']:.2%}, bike {target['bike']:.2%}")

    asc = {m: 0.0 for m in ACTIVE}
    history = []
    for it in range(max_iter):
        for s in sims:
            s._mode_asc = dict(asc)
        sums, count = {}, 0
        for s, grp in zip(sims, groups):
            sh = expected_shares(s, grp)
            for m, v in sh.items():
                sums[m] = sums.get(m, 0.0) + v * len(grp)
            count += len(grp)
        sim_sh = {m: v / count for m, v in sums.items()}
        history.append((dict(asc), dict(sim_sh)))
        gap = max(abs(sim_sh.get(m, 0.0) - target[m]) for m in ACTIVE)
        log(f"  iter {it:2d}: ASC walk {asc['walk']:+.3f} bike {asc['bike']:+.3f} -> "
            f"walk {sim_sh.get('walk', 0):.2%} bike {sim_sh.get('bike', 0):.2%} "
            f"car {sim_sh.get('car', 0):.2%}")
        if gap < tol:
            break
        for m in ACTIVE:
            obs, got = target[m], sim_sh.get(m, 0.0)
            if obs <= 0 or got <= 0:
                continue
            asc[m] = max(-15.0, min(15.0, asc[m] + math.log(obs / got)))
    return {"asc": asc, "target": target, "simulated": history[-1][1], "n": n,
            "converged": gap < tol, "single": {m: c / n_total for m, c in only.items()}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("metros", nargs="*")
    ap.add_argument("--agents", type=int, default=6000)
    ap.add_argument("--replicates", type=int, nargs="+", default=[0, 1])
    args = ap.parse_args()
    metros = args.metros or list(params.METRO_PARAMS["metros"])
    results = {}
    for metro in metros:
        t0 = time.time()
        results[metro] = calibrate(metro, args.agents, args.replicates)
        print(f"  {metro} done in {time.time() - t0:.0f}s", flush=True)
    print("\nMODE_ASC = {")
    for metro, r in results.items():
        print(f'    "{metro}": {{"walk": {r["asc"]["walk"]:.3f}, "bike": {r["asc"]["bike"]:.3f}}},'
              f'  # n={r["n"]:,} observed walk {r["target"]["walk"]:.2%} bike '
              f'{r["target"]["bike"]:.2%}; fitted walk {r["simulated"].get("walk", 0):.2%} '
              f'bike {r["simulated"].get("bike", 0):.2%}'
              + ('; single-option ' + ", ".join(f"{m} {v:.1%}" for m, v in sorted(r["single"].items()))
                 if r["single"] else "")
              + f'{"" if r["converged"] else "  NOT CONVERGED"}')
    print("}")


if __name__ == "__main__":
    main()

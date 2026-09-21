#!/usr/bin/env python3
"""Run a tiny simulation with no database and no survey data.

    python3 scripts/run_demo.py

This is the "does it work at all" path for someone who has just cloned the
repo. It downloads a small OpenStreetMap network once (lower Manhattan, cached
afterwards), builds agents from the FABRICATED personas written by
data/make_demo_personas.py, runs a few days and prints what happened.

IT REPRODUCES NOTHING. The personas are invented, there is no census
population behind the agents, and the venue catalog is synthetic unless you
supply your own POI data. The real pipeline — LODES commutes, ACS/PUMS
demographics and the survey population — is described in
docs/acs-population.md and needs the data those scripts download.
"""
from __future__ import annotations

import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from welfare_rs import Simulation, build_road_network  # noqa: E402

PERSONAS = os.path.join(REPO, "data", "demo", "demo_personas_SYNTHETIC.csv")


def main() -> int:
    if not os.path.exists(PERSONAS):
        print("No demo personas yet. Run:  python3 data/make_demo_personas.py")
        return 1

    print("building the road network (first run downloads from OpenStreetMap)...")
    net = build_road_network("nyc_lower_manhattan")
    print(f"  {net}")

    sim = Simulation(
        num_agents=50,
        seed=1,
        road_network=net,
        persona_csv_path=PERSONAS,
    )
    print(f"  {len(sim.agents)} agents, {len(sim.place_catalog)} places")

    sim.run_days(3)
    stats = sim.stats or {}
    print("\nafter 3 days:")
    for key in sorted(stats):
        print(f"  {key}: {stats[key]}")
    print("\nReminder: these agents are fabricated. Nothing here reproduces a result.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

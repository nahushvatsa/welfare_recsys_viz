"""End-to-end verification of the LODES commute chain on one warmed metro.

Proves, in order: graph loads -> commute pairs attach -> agents get REAL
home/work -> workplaces are inside the core -> a day actually simulates.
Exits non-zero on the first failed assertion.
"""
import sys, time, random
import numpy as np

metro = sys.argv[1] if len(sys.argv) > 1 else "miami"
fail = []

def check(label, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}{('  ' + detail) if detail else ''}")
    if not ok:
        fail.append(label)

from welfare_rs import params
from welfare_rs.metro import build_metro_network, excluded_counties
from welfare_rs.simulation import Simulation
from welfare_rs.utils import haversine_km
from welfare_rs.datasource import get_datasource, select_counties

print(f"\n=== 1. graph loads from warm cache ({metro}) ===")
t0 = time.time()
net = build_metro_network(metro)
print(f"  loaded in {time.time()-t0:.1f}s")
check("graph has nodes", net.num_base_nodes > 1000, f"{net.num_base_nodes:,} intersections")
check("graph has a core layer", net.has_layers and bool(net._core_nodes),
      f"{len(net._core_nodes or []):,} core nodes")
check("edges carry travel_time", getattr(net, "_has_edge_times", False))

print("\n=== 2. LODES commute pairs attached ===")
check("has_commutes", net.has_commutes)
check("pair count > 0", net.num_commute_pairs > 0, f"{net.num_commute_pairs:,} pairs")

ds = get_datasource()
selected = select_counties(ds.county_flows(metro))
dropped = excluded_counties(metro)
counties = [c for c in selected if c.fips not in dropped]
note = f" ({len(dropped)} excluded as disconnected)" if dropped else ""
print(f"  graph covers {len(counties)} counties (90% rule){note}")

print("\n=== 3. agents get REAL home/work from those pairs ===")
sim = Simulation(num_agents=200, seed=42, road_network=net,
                 poi_rows=ds.poi_rows(metro), disabled_modes=("transit",))
homes = {a.home for a in sim.agents}
works = {a.work for a in sim.agents}
check("homes are varied", len(homes) > 50, f"{len(homes)} distinct of {len(sim.agents)}")
check("workplaces are varied", len(works) > 20, f"{len(works)} distinct")

# The invariant belongs on the DATA, not on the snapped node. Every LODES work
# block must sit inside the core polygon — that is what "workplaces live in the
# core" means, and it is checked exhaustively below. The graph node a block
# snaps to may legitimately fall just outside: LA city takes in a long stretch
# of the Santa Monica Mountains where a block's nearest paved intersection is
# across the city line. Snapping there is correct — it IS the closest road, and
# the gap is charged as an access leg — so a hard failure on the node would be
# rejecting the right answer. Asserted on blocks, bounded on nodes.
if net.has_commutes:
    c = net._commutes
    blocks = {(round(float(a_), 6), round(float(o_), 6))
              for a_, o_ in zip(c.w_lat, c.w_lon)}
    out_blocks = [b for b in blocks if not net.in_core(*b)]
    check("ALL work BLOCKS inside the core polygon", not out_blocks,
          f"{len(blocks) - len(out_blocks):,}/{len(blocks):,} distinct blocks")

in_core = [net.in_core(*a.work) for a in sim.agents]
off = [a.work_access_km for a, ok in zip(sim.agents, in_core) if not ok]
check("snapped workplaces are in or beside the core",
      sum(in_core) >= 0.98 * len(in_core) and (not off or max(off) < 5.0),
      f"{sum(in_core)}/{len(in_core)} land on a core node"
      + (f"; {len(off)} snapped outside, max {max(off):.2f} km" if off else ""))

home_in_core = [net.in_core(*a.home) for a in sim.agents]
print(f"  homes inside core: {sum(home_in_core)}/{len(home_in_core)} "
      f"({100*sum(home_in_core)/len(home_in_core):.0f}%) — rest commute in from the shell")

d = [haversine_km(a.home, a.work) for a in sim.agents]
print(f"  commute distance km: min {min(d):.1f}  median {np.median(d):.1f}  "
      f"mean {np.mean(d):.1f}  max {max(d):.1f}")
check("commute distances are plausible", 1.0 < np.median(d) < 60.0)

print("\n=== 4. snap distance (block point -> graph node) ===")
acc_h = np.array([a.home_access_km for a in sim.agents])
acc_w = np.array([a.work_access_km for a in sim.agents])
print(f"  home access km: median {np.median(acc_h):.3f}  p90 {np.percentile(acc_h,90):.3f}  max {acc_h.max():.3f}")
print(f"  work access km: median {np.median(acc_w):.3f}  p90 {np.percentile(acc_w,90):.3f}  max {acc_w.max():.3f}")
check("home snap is bounded", acc_h.max() < 25.0, "sanity ceiling")
check("work snap is small (core is full-detail)", np.median(acc_w) < 0.5)

print("\n=== 5. a day actually simulates ===")
t0 = time.time()
sim.run_days(1)
s = sim.summarize()
check("trips were made", s["trips"] > 0, f"{s['trips']} trips in {time.time()-t0:.1f}s")
check("travel time is sane", 1 < s["avg_travel_time_min"] < 240,
      f"avg {s['avg_travel_time_min']:.1f} min")
leisure = sum(1 for a in sim.agents for t in a.trips if t.purpose == "leisure")
print(f"  leisure trips: {leisure}   POIs in catalog: {len(sim.place_catalog):,}")

print("\n" + ("ALL CHECKS PASSED" if not fail else f"FAILED: {fail}"))
sys.exit(1 if fail else 0)

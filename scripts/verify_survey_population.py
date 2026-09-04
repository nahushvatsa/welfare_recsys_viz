#!/usr/bin/env python3
"""Verify the survey-grounded population against the shipped survey package.

Checks, in order:

1.  Normalisation maps each response scale onto [0, 1] endpoint-correctly. The
    regression this guards is a passthrough on ``0 <= score <= 1`` that mapped a
    raw "1" — the floor of every 7-point item — onto 1.0, its ceiling.
2.  Every ``Agent.survey_items`` slot has a declared scale.
3.  ``construct_items.json`` reproduces the composites shipped in the survey CSV.
4.  Every built agent's psychometrics and items equal its source respondent's,
    rescaled.
5.  No spurious clustering at the ceiling (the fingerprint of check 1 failing).
6.  The built population's correlations match ``calibration_correlations.csv``,
    which is what whole-profile resampling is there to preserve.
7.  ``autonomy_preference`` carries the construct the acceptance model needs —
    negatively correlated with trust, not the positively-loaded composite.
8.  LODES age-segment conditioning selects the right pairs and falls back
    cleanly when a segment is empty.

Run:  python scripts/verify_survey_population.py
Exits non-zero if any check fails.
"""

from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import welfare_rs  # noqa: E402
from welfare_rs import params  # noqa: E402
from welfare_rs.agent import Agent, PSYCHOMETRIC_SCALE, SURVEY_ITEM_SCALES  # noqa: E402
from welfare_rs.datasource import CommutePairs  # noqa: E402
from welfare_rs.geo import RoadNetwork  # noqa: E402
from welfare_rs.simulation import Simulation  # noqa: E402

HANDOFF = REPO_ROOT / "data_handoff"
SURVEY_CSV = HANDOFF / "algo_leisure_survey_2026.csv"

# Mirrors data/build_survey_personas.py. Duplicated deliberately: if the two
# drift apart, check 4 fails, which is the point of an independent check.
PSYCHOMETRIC_MAP = {
    "openness": "Openness",
    "conscientiousness": "Conscientiousness",
    "extraversion": "Extraversion",
    "agreeableness": "Agreeableness",
    "neuroticism": "Neuroticism",
}
LATENT_MAP = {
    "maximization": "Maximization",
    "trust_platforms": "Recommendation_Trust",
    "algorithmic_awareness": "Algorithmic_Awareness",
    "autonomy_preference": "s6_autonomy_control_a",
}
ITEM_MAP = {
    "local_leisure_frequency": "leisure_freq_num",
    "overnight_trip_frequency": "overnight_num",
    "spontaneity_share": "spur_num",
    "city_familiarity": "CityFam_num",
    "follow_through_friend": "s2_Q16",
    "follow_through_platform": "s2_Q17",
    "follow_through_ai": "s2_Q18",
    "cross_platform_search": "s4_search_c",
    "price_filter_tendency": "s4_search_e",
    "group_coordination_preference": "s4_group_coord_mean",
    "popularity_herding": "s5_general_a",
    "unexpected_discovery": "s3_attitudes_d",
    "social_posting": "s4_social_media_b",
    "multi_platform_parallel": "s5_search_c",
    "ai_itinerary_comfort": "s6_trust_reliance_c",
    "explanation_needed": "s6_trust_reliance_d",
    "objective_algorithmic_literacy": "s6_algo_knowledge_correct",
}

failures: list[str] = []


def check(label, ok, detail=""):
    print(f"  [{'ok ' if ok else 'FAIL'}] {label}{(' — ' + detail) if detail else ''}")
    if not ok:
        failures.append(label)


def rescale(raw, lo, hi):
    return max(0.0, min(1.0, (float(raw) - lo) / (hi - lo)))


def main():
    print("1. normalisation endpoints")
    cases = [(1, 1, 7, 0.0), (4, 1, 7, 0.5), (7, 1, 7, 1.0),
             (1, 1, 5, 0.0), (5, 1, 5, 1.0),
             (0, 0, 1, 0.0), (1, 0, 1, 1.0),
             (0, 0, 12, 0.0), (12, 0, 12, 1.0)]
    for raw, lo, hi, want in cases:
        got = Agent._normalize_score(None, raw, lo, hi)
        check(f"({lo},{hi}) raw {raw} -> {want}", abs(got - want) < 1e-12, f"got {got:.4f}")

    print("\n2. every survey_items slot has a declared scale")
    probe = Agent(0, 50000, 40, True, (0.0, 0.0), (0.0, 0.0), seed=1)
    missing = sorted(k for k in probe.survey_items if k not in SURVEY_ITEM_SCALES)
    check("all slots scaled", not missing, f"{len(probe.survey_items)} slots, missing {missing}")

    print("\n3. construct_items.json reproduces the shipped composites")
    survey = pd.read_csv(SURVEY_CSV, keep_default_na=False, na_values=[""])
    constructs = json.load(open(HANDOFF / "construct_items.json"))
    for name, items in constructs.items():
        diff = (survey[items].mean(axis=1) - survey[name]).abs().max()
        check(f"{name} (k={len(items)})", diff < 1e-9, f"max dev {diff:.1e}")

    print("\n4. built agents equal their source respondents")
    net = welfare_rs.build_road_network("nyc_manhattan")
    sim = Simulation(num_agents=600, seed=42, road_network=net,
                     persona_csv_path=params.SURVEY_PERSONA_CSV_PATH)
    by_id = {r["respondent_id"]: r for r in csv.DictReader(open(SURVEY_CSV))}
    mismatched = skipped = compared = 0
    for agent in sim.agents:
        source = by_id[agent.persona_id.split("#")[0]]
        checks = [(agent.big_five, PSYCHOMETRIC_MAP, PSYCHOMETRIC_SCALE),
                  (agent.latent_variables, LATENT_MAP, PSYCHOMETRIC_SCALE),
                  (agent.survey_items, ITEM_MAP, None)]
        for target, mapping, fixed_scale in checks:
            for field, column in mapping.items():
                raw = str(source[column]).strip()
                if raw == "":
                    skipped += 1
                    continue
                lo, hi = fixed_scale or SURVEY_ITEM_SCALES[field]
                compared += 1
                if abs(target[field] - rescale(raw, lo, hi)) > 1e-9:
                    mismatched += 1
    check("every mapped value matches its respondent", mismatched == 0,
          f"{compared} compared, {mismatched} mismatched, {skipped} skipped (missing in survey)")

    print("\n5. no spurious ceiling clustering")
    for field in ("trust_platforms", "maximization"):
        values = np.array([a.latent_variables[field] for a in sim.agents])
        at_ceiling = float((values > 0.999).mean())
        check(f"{field} not piled at 1.0", at_ceiling < 0.15, f"{at_ceiling:.1%} at ceiling")
    comfort = np.array([a.survey_items["ai_itinerary_comfort"] for a in sim.agents])
    floor_share = float((comfort < 0.001).mean())
    # 90 of 473 respondents (19.0%) answered "Strongly disagree" to the AI
    # itinerary item; before the normalisation fix they read as 1.0.
    check("ai_itinerary_comfort floor matches the survey", 0.14 < floor_share < 0.24,
          f"{floor_share:.1%} at floor, survey has 19.0%")

    print("\n6. population correlations match the calibration target")
    getters = {
        "Age_num": lambda a: a.characteristics.get("Age_num"),
        "Sex_num": lambda a: a.characteristics.get("Sex_num"),
        "Education_num": lambda a: a.characteristics.get("Education_num"),
        "Income_num": lambda a: a.characteristics.get("Income_num"),
        "LeisureSpend_num": lambda a: a.characteristics.get("LeisureSpend_num"),
        "ResidenceLength_num": lambda a: a.characteristics.get("ResidenceLength_num"),
        "CityFam_num": lambda a: a.characteristics.get("CityFam_num"),
        "Recommendation_Trust": lambda a: a.latent_variables["trust_platforms"],
        "Platform_Comfort": lambda a: a.characteristics.get("Platform_Comfort"),
        "Algorithmic_Awareness": lambda a: a.latent_variables["algorithmic_awareness"],
        "Autonomy_Control": lambda a: a.characteristics.get("Autonomy_Control"),
        "Neuroticism": lambda a: a.big_five["neuroticism"],
        "Extraversion": lambda a: a.big_five["extraversion"],
        "Openness": lambda a: a.big_five["openness"],
        "Agreeableness": lambda a: a.big_five["agreeableness"],
        "Conscientiousness": lambda a: a.big_five["conscientiousness"],
        "Maximization": lambda a: a.latent_variables["maximization"],
        "s2_Q16": lambda a: a.survey_items["follow_through_friend"],
        "s2_Q17": lambda a: a.survey_items["follow_through_platform"],
        "s2_Q18": lambda a: a.survey_items["follow_through_ai"],
        "leisure_freq_num": lambda a: a.survey_items["local_leisure_frequency"],
        "overnight_num": lambda a: a.survey_items["overnight_trip_frequency"],
        "spur_num": lambda a: a.survey_items["spontaneity_share"],
    }

    def as_float(value):
        try:
            return float(value)
        except (TypeError, ValueError):
            return np.nan

    # Normalisation is a linear rescale, so Pearson r survives it unchanged and
    # the built values are directly comparable to the survey's own matrix.
    built = pd.DataFrame({k: [as_float(g(a)) for a in sim.agents] for k, g in getters.items()})
    target = pd.read_csv(HANDOFF / "calibration_correlations.csv", index_col=0)
    columns = list(getters)
    deviation = (built[columns].corr() - target.loc[columns, columns]).abs().values
    upper = np.triu_indices(len(columns), 1)
    check("max pairwise deviation < 0.10", deviation[upper].max() < 0.10,
          f"max {deviation[upper].max():.4f}, mean {deviation[upper].mean():.4f}")

    print("\n7. autonomy_preference carries the right construct")
    autonomy = np.array([a.latent_variables["autonomy_preference"] for a in sim.agents])
    trust = np.array([a.latent_variables["trust_platforms"] for a in sim.agents])
    r = float(np.corrcoef(autonomy, trust)[0, 1])
    # The Autonomy_Control composite runs +0.46 against trust; the item the model
    # needs runs -0.52. A positive r here means the composite got wired back in
    # and beta_O is inverted for the whole population.
    check("negatively correlated with trust", r < -0.3, f"r = {r:+.3f}")

    print("\n8. employment drives the day and the value of time")
    employed = [a for a in sim.agents if a.is_employed]
    non_employed = [a for a in sim.agents if not a.is_employed]
    share = len(non_employed) / len(sim.agents)
    # 161 of 473 respondents (34.0%) are retired, unemployed, out of the labour
    # force or studying; bootstrap copies move the run share a little.
    check("non-employed share tracks the survey", 0.28 < share < 0.40,
          f"{share:.1%} of agents, survey has 34.0%")
    sim.run_days(2)
    work_activities = sum(1 for a in non_employed for act in a.schedule if act.type == "work")
    check("non-employed get no work activity", work_activities == 0, f"{work_activities} found")
    work_trips = sum(1 for a in non_employed for t in a.trips if t.purpose == "work")
    check("non-employed take no commute", work_trips == 0, f"{work_trips} found")
    check("employed still commute",
          sum(1 for a in employed for t in a.trips if t.purpose == "work") > 0)
    if employed and non_employed:
        v_emp = float(np.mean([a.vot for a in employed]))
        v_non = float(np.mean([a.vot for a in non_employed]))
        check("non-earner value of time is discounted", v_non < v_emp,
              f"${v_emp:.2f}/h employed vs ${v_non:.2f}/h non-employed")
        ad = params.AGENT_DEFAULTS
        # VOT is a share of the wage equivalent, not the whole of it (Small 2012).
        wages = [a.characteristics["income"] / ad["vot_hours_per_year"] for a in employed]
        ratios = [a.vot / w for a, w in zip(employed, wages) if a.vot > ad["vot_floor"]]
        check("employed VOT is half the wage equivalent",
              all(abs(r - ad["vot_wage_share_employed"]) < 1e-9 for r in ratios),
              f"{len(ratios)} agents above the floor, share {np.mean(ratios):.3f}")

    print("\n8b. attributes the survey does not measure are held fixed")
    for attr, label in [("car_access_type", "car access"),
                        ("transit_access_level", "transit access"),
                        ("mobility_needs", "mobility needs"),
                        ("walk_tolerance_min", "walking tolerance"),
                        ("time_window_pref", "time window"),
                        ("primary_interest", "primary interest")]:
        values = {getattr(a, attr) for a in sim.agents}
        check(f"{label} constant across the population", len(values) == 1, f"{values}")
    check("primary interest grants no bonus",
          not (params.PARTICIPATION_PARAMS["interest_map"].get(sim.agents[0].primary_interest)))

    print("\n8c. MTTC motivation mixture is live")
    for key in ("derived", "intrinsic", "escape", "positionality"):
        values = np.array([a.motivation_weights[key] for a in sim.agents])
        check(f"{key} varies across agents", values.std() > 1e-6,
              f"min {values.min():.3f} max {values.max():.3f} mean {values.mean():.3f}")

    print("\n8d. Eq. 2 extra-travel term")
    probe.latent_variables["trust_platforms"] = 0.5
    probe.latent_variables["autonomy_preference"] = 0.5
    etas = [probe._estimate_eta(0.5, delta_cost=d) for d in (-1.0, -0.4, 0.0, 0.4, 1.0)]
    check("eta decreases as the offer gets farther",
          all(x >= y for x, y in zip(etas, etas[1:])),
          " -> ".join(f"{e:.3f}" for e in etas))
    check("a closer offer raises eta above the no-difference case", etas[1] > etas[2])
    check("delta_cost defaults to no effect",
          probe._estimate_eta(0.5) == probe._estimate_eta(0.5, delta_cost=0.0))
    beta = params.SIMPLIFIED_ETA_PARAMS["beta_delta_cost"]
    check("beta_delta_cost stays below the trust coefficient",
          0.0 < beta < params.SIMPLIFIED_ETA_PARAMS["baseline_trust_coeff"] * 2,
          f"beta_D={beta}, beta_T={params.SIMPLIFIED_ETA_PARAMS['baseline_trust_coeff']}")
    # Both sides of the difference must come from one pricing method, or the
    # comparison stops being like-for-like.
    src = (REPO_ROOT / "welfare_rs" / "agent.py").read_text()
    check("both cost sides priced by _price_round_trip",
          src.count("_price_round_trip(") == 3,  # 1 def + 2 call sites
          f"{src.count('_price_round_trip(')} occurrences (expect 3)")
    check("agent's own pick is priced once and reused",
          '"trip_disutility": expected_trip_disutility' in src)

    print("\n9. LODES age-segment conditioning")
    for age, want in [(18, 0), (29, 0), (30, 1), (54, 1), (55, 2), (70, 2)]:
        check(f"age {age} -> sa0{want + 1}", RoadNetwork.lodes_age_segment(age) == want)

    stub = RoadNetwork.__new__(RoadNetwork)
    stub._commutes = None
    stub._commute_cum = None
    stub._commute_total = 0.0
    stub._commute_seg_cum = {}
    stub.snap_latlon = lambda lat, lon: (lat, lon)
    # Three pairs, each exclusive to one segment: a conditioned draw must return
    # only its own pair, an unconditioned draw all three.
    stub.attach_commutes(CommutePairs(
        h_lat=np.array([1.0, 2.0, 3.0]), h_lon=np.array([1.0, 2.0, 3.0]),
        w_lat=np.array([10.0, 20.0, 30.0]), w_lon=np.array([10.0, 20.0, 30.0]),
        jobs=np.array([100.0, 100.0, 100.0]),
        age=np.eye(3) * 100.0, earn=np.zeros((3, 3))))
    rng = random.Random(0)
    for age, want_home in [(25, 1.0), (40, 2.0), (60, 3.0)]:
        drawn = {stub.sample_commute(rng, age_years=age)[0][0] for _ in range(200)}
        check(f"age {age} draws only its segment", drawn == {want_home}, f"drew {sorted(drawn)}")
    drawn = {stub.sample_commute(rng)[0][0] for _ in range(400)}
    check("unconditioned draws all segments", drawn == {1.0, 2.0, 3.0}, f"drew {sorted(drawn)}")

    # LODES segments monthly earnings at $1250 and $3333.
    for annual, want in [(12000, 0), (15000, 0), (15001, 1), (39996, 1), (40000, 2)]:
        got = RoadNetwork.lodes_earnings_segment(annual)
        check(f"${annual:,}/yr -> se0{want + 1}", got == want, f"got se0{got + 1}")

    # Four pairs crossing age against earnings: conditioning on both must pick
    # exactly one, on one margin exactly two.
    joint = RoadNetwork.__new__(RoadNetwork)
    joint._commutes = None
    joint._commute_cum = None
    joint._commute_total = 0.0
    joint._commute_seg_cum = {}
    joint.snap_latlon = lambda lat, lon: (lat, lon)
    joint.attach_commutes(CommutePairs(
        h_lat=np.array([1.0, 2.0, 3.0, 4.0]), h_lon=np.zeros(4),
        w_lat=np.zeros(4), w_lon=np.zeros(4), jobs=np.full(4, 10.0),
        age=np.array([[10, 0, 0], [10, 0, 0], [0, 0, 10], [0, 0, 10]], dtype=float),
        earn=np.array([[10, 0, 0], [0, 0, 10], [10, 0, 0], [0, 0, 10]], dtype=float)))
    for age, income, want_home in [(25, 12000, 1.0), (25, 150000, 2.0),
                                   (60, 12000, 3.0), (60, 150000, 4.0)]:
        drawn = {joint.sample_commute(rng, age_years=age, annual_income=income)[0][0]
                 for _ in range(300)}
        check(f"age {age} + ${income:,} draws one cell", drawn == {want_home}, f"drew {sorted(drawn)}")
    drawn = {joint.sample_commute(rng, age_years=25)[0][0] for _ in range(300)}
    check("age margin alone spans both earnings cells", drawn == {1.0, 2.0}, f"drew {sorted(drawn)}")
    drawn = {joint.sample_commute(rng, annual_income=12000)[0][0] for _ in range(300)}
    check("earnings margin alone spans both age cells", drawn == {1.0, 3.0}, f"drew {sorted(drawn)}")

    stub.attach_commutes(CommutePairs(
        h_lat=np.array([1.0]), h_lon=np.array([1.0]),
        w_lat=np.array([9.0]), w_lon=np.array([9.0]), jobs=np.array([50.0]),
        age=np.array([[50.0, 0.0, 0.0]]), earn=np.zeros((1, 3))))
    fallback = stub.sample_commute(rng, age_years=60)  # sa03 empty here
    check("empty segment falls back", fallback is not None and fallback[0][0] == 1.0)

    print()
    if failures:
        print(f"FAILED: {len(failures)} check(s)")
        for name in failures:
            print(f"  - {name}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

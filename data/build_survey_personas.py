#!/usr/bin/env python3
"""Build the ABM's persona file from the Prolific stated-preference survey.

Source: ``data_handoff/algo_leisure_survey_2026.csv`` — the April 2026 survey
run under IRB-FY2026-11354 (N=473 after consent, completion and four attention
checks). Output: ``data/survey_personas_2026.csv``, one row per respondent, in
the column layout :meth:`welfare_rs.simulation.Simulation._load_personas` reads.

Each respondent becomes one agent profile, so the joint distribution over
personality, platform attitudes and demographics is the observed one rather than
a product of marginals.

Two classes of column are emitted:

1. **Measured** — psychometric composites and survey items, written on their
   native response scale (7-point Likert as 1..7, not pre-normalised). The
   consumer normalises against ``welfare_rs.agent.SURVEY_ITEM_SCALES``.
2. **Derived** — persona fields the survey determines indirectly (``Budget``
   from monthly leisure spend, ``TopRated`` from stated popularity-seeking).
Fields the survey does not collect (car and transit access, walking tolerance,
mobility needs, primary leisure interest, preferred time window, travel-party
composition, environmental attitude, risk salience) are not emitted: the
simulation holds them fixed for every agent. See the note below.

Run:  python data/build_survey_personas.py
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SURVEY_CSV = REPO_ROOT / "data_handoff" / "algo_leisure_survey_2026.csv"
DEFAULT_OUT_CSV = REPO_ROOT / "data" / "survey_personas_2026.csv"


# ── Measured columns: canonical agent field -> survey column ─────────────────
#
# Psychometric composites, all unweighted means of 7-point items.
PSYCHOMETRIC_MAP = {
    "openness": "Openness",
    "conscientiousness": "Conscientiousness",
    "extraversion": "Extraversion",
    "agreeableness": "Agreeableness",
    "neuroticism": "Neuroticism",
    "maximization": "Maximization",
    "trust_platforms": "Recommendation_Trust",
    "algorithmic_awareness": "Algorithmic_Awareness",
    # NOT Autonomy_Control. That composite is positively loaded on liking
    # personalisation (+0.46 with Recommendation_Trust, +0.43 with following
    # platform recommendations), while this slot enters the acceptance
    # probability with a negative coefficient. s6_autonomy_control_a — "I prefer
    # to make my own leisure choices rather than follow platform suggestions" —
    # is the item that measures the construct the ABM needs (-0.52 with trust)
    # and is deliberately excluded from every composite in the survey package.
    "autonomy_preference": "s6_autonomy_control_a",
}

# Survey items, on their native scale. Slots absent here have no counterpart in
# this survey and keep the agent default of 0.5: incentive_coupon,
# incentive_sponsored, incentive_loyalty (no incentive battery), review_posting,
# switch_when_dissatisfied, advice_goal_directed. See docs/survey-coverage.md for
# what each one damps and why it was kept rather than dropped.
ITEM_MAP = {
    "local_leisure_frequency": "leisure_freq_num",        # outings/week, 0.5-6.0
    "overnight_trip_frequency": "overnight_num",          # trips/12mo, 0-12
    "spontaneity_share": "spur_num",                      # 1-5
    "city_familiarity": "CityFam_num",                    # 1-5
    "follow_through_friend": "s2_Q16",                    # 1-7
    "follow_through_platform": "s2_Q17",                  # 1-7
    "follow_through_ai": "s2_Q18",                        # 1-7
    "cross_platform_search": "s4_search_c",               # 1-7
    "price_filter_tendency": "s4_search_e",               # 1-7, 15 missing
    "group_coordination_preference": "s4_group_coord_mean",  # 1-7
    "popularity_herding": "s5_general_a",                 # 1-7
    "unexpected_discovery": "s3_attitudes_d",             # 1-7
    "social_posting": "s4_social_media_b",                # 1-7
    "multi_platform_parallel": "s5_search_c",             # 1-7
    "ai_itinerary_comfort": "s6_trust_reliance_c",        # 1-7
    "explanation_needed": "s6_trust_reliance_d",          # 1-7
    "objective_algorithmic_literacy": "s6_algo_knowledge_correct",  # 0/1
}

# ── Carried columns: recorded on the agent, no behavioural effect ────────────
CARRIED_COLUMNS = (
    "Education",
    "Education_num",
    "Employment",
    "RaceEthnicity",
    "HomeLanguage",
    "CensusRegion",
    "CensusDivision",
    "ResidenceLength",
    "ResidenceLength_num",
    "LeisureSpend",
    "LeisureSpend_num",
    "CityFam_num",
    "Age_num",
    "Income_num",
    "Sex_num",
    "scenario",
    # Retained for analysis only. Not wired to autonomy_preference — see above.
    "Autonomy_Control",
    "Platform_Comfort",
)

# ── Not collected by the survey ─────────────────────────────────────────────
#
# Car and transit access, mobility needs, walking tolerance, primary leisure
# interest, preferred time window, travel-party composition, environmental
# attitude and risk salience have no counterpart anywhere in this instrument
# (checked against every one of the 136 columns). They used to be drawn per
# agent from the synthetic persona file, whose level shares came from a
# D-optimal experimental design — balanced by construction, not representative
# of any population — which made them look like a source of empirical
# heterogeneity when they were an artefact of the design.
#
# They are no longer emitted at all. The simulation holds them fixed for the
# whole population (welfare_rs.params.PERSONA_MAPPING["fixed_attributes"]), and
# primary interest and environmental attitude are dropped outright rather than
# pinned, since a constant utility bonus for every agent is only a shift in
# origin. Every column in this file is now survey-derived.


def _to_float(value):
    if value is None:
        return None
    token = str(value).strip()
    if token == "":
        return None
    try:
        return float(token)
    except ValueError:
        return None


def _median(values):
    ordered = sorted(values)
    if not ordered:
        return None
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def build_rows(survey_rows):
    """Return (persona_rows, notes) for the shipped survey rows."""
    # Monthly leisure spend drives both the Budget label and budget_tightness.
    # Five respondents chose "I'm not sure" and are left missing in the shipped
    # file; they take the sample median so neither field silently defaults.
    spend_values = [
        v for v in (_to_float(r.get("LeisureSpend_num")) for r in survey_rows) if v is not None
    ]
    spend_median = _median(spend_values)
    n_spend_imputed = 0

    personas = []
    for row in survey_rows:
        persona = {"PersonaID": row["respondent_id"]}

        # Demographics the agent builder reads. Band labels are looked up in
        # params.PERSONA_MAPPING, which carries the survey's own band strings.
        persona["Age"] = row["Age"]
        persona["Income"] = row["HouseholdIncome"]
        persona["Sex"] = row["Sex"]

        spend = _to_float(row.get("LeisureSpend_num"))
        if spend is None:
            spend = spend_median
            n_spend_imputed += 1
        # budget_tightness is a [0, 1] proportion: 1 = tightest (< $50/month).
        persona["budget_tightness"] = round(1.0 - (spend - 1.0) / 4.0, 6)
        persona["Budget"] = "low" if spend <= 2 else ("medium" if spend <= 3 else "high")

        # TopRated sets the base level of practice conformity, which the survey's
        # popularity-seeking item then modulates through trend susceptibility.
        # Deriving it from the same item keeps base and shift pointing the same
        # way; drawing it independently would inject noise uncorrelated with the
        # respondent's own stated preference for well-known destinations.
        popularity = _to_float(row.get("s5_general_a"))
        persona["TopRated"] = "yes" if (popularity is not None and popularity >= 5) else "no"

        for field, column in PSYCHOMETRIC_MAP.items():
            persona[field] = row[column]
        for field, column in ITEM_MAP.items():
            persona[field] = row.get(column, "")

        for column in CARRIED_COLUMNS:
            persona[column] = row.get(column, "")

        # The survey is a national sample with home ZIP removed for
        # de-identification, so it carries no usable geography. Homes and
        # workplaces come from the metro's LODES commute pairs; on an unlayered
        # single-city network the loader falls back to a random network node.
        persona["start_latitude"] = ""
        persona["start_longitude"] = ""

        personas.append(persona)

    notes = {"n_spend_imputed": n_spend_imputed, "spend_median": spend_median}
    return personas, notes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--survey", type=Path, default=DEFAULT_SURVEY_CSV)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_CSV)
    args = parser.parse_args()

    # keep_default_na equivalent: csv gives strings, so the pandas "None" ->
    # NaN trap documented in the handoff README cannot bite here.
    with args.survey.open(newline="", encoding="utf-8") as f:
        survey_rows = list(csv.DictReader(f))
    if not survey_rows:
        raise SystemExit(f"no rows read from {args.survey}")

    personas, notes = build_rows(survey_rows)

    fieldnames = list(personas[0].keys())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(personas)

    print(f"wrote {len(personas)} personas x {len(fieldnames)} columns -> {args.out}")
    if notes["n_spend_imputed"]:
        print(
            f"  LeisureSpend_num imputed at the sample median "
            f"({notes['spend_median']}) for {notes['n_spend_imputed']} respondents"
        )


if __name__ == "__main__":
    main()

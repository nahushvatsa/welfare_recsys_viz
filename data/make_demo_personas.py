#!/usr/bin/env python3
"""Generate a FABRICATED persona file so the engine can be run without the survey.

    python3 data/make_demo_personas.py --n 200

The real population comes from a Prolific survey held under IRB-FY2026-11354.
It may not leave institutional storage, so a clone of this repo has no persona
file and the engine cannot start. This writes a stand-in with the same columns
and the same value ranges, filled with numbers drawn from thin air.

NOTHING HERE IS DATA. Every row is invented. The file exists so someone can run
a simulation end to end and see the machinery work — never to reproduce a
result. Every PersonaID is prefixed ``SYNTHETIC-``, and
``welfare_rs.simulation`` warns loudly when it sees that prefix, so a run built
on this file announces itself in its own logs.

Item ranges come from ``welfare_rs.agent.SURVEY_ITEM_SCALES`` rather than being
restated here, so a fabricated value can never fall outside the scale its
consumer normalises against.
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from welfare_rs.agent import PSYCHOMETRIC_SCALE, SURVEY_ITEM_SCALES  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_survey_personas import ITEM_MAP  # noqa: E402

# Only the slots the real survey file carries. The seven the survey cannot
# measure are deliberately left out, so a demo agent hits the same 0.5 default
# the real population does instead of a fabricated value — the demo should
# exercise the real code path, not a more generous one.
DEMO_ITEMS = {name: SURVEY_ITEM_SCALES[name] for name in ITEM_MAP
              if name in SURVEY_ITEM_SCALES}

DEFAULT_OUT = os.path.join(REPO, "data", "demo", "demo_personas_SYNTHETIC.csv")

PSYCHOMETRICS = ["openness", "conscientiousness", "extraversion", "agreeableness",
                 "neuroticism", "maximization", "trust_platforms",
                 "algorithmic_awareness", "autonomy_preference"]

AGE_BANDS = ["18-24", "25-34", "35-44", "45-54", "55-64", "65 or older"]
INCOME_BANDS = ["Under $25,000", "$25,000 - $49,999", "$50,000 - $74,999",
                "$75,000 - $99,999", "$100,000 - $149,000",
                "$150,000 - $199,999", "$200,000 or more"]
EMPLOYMENT = ["Employed full-time", "Employed part-time", "Self-employed",
              "Retired", "Unemployed and looking for work",
              "Not in the labor force (e.g., homemaker, caregiver)", "Student"]
CARRIED = {
    "Education": "Bachelor's degree", "Education_num": "4",
    "RaceEthnicity": "Not reported", "HomeLanguage": "English",
    "CensusRegion": "Northeast", "CensusDivision": "Middle Atlantic",
    "ResidenceLength": "3-5 years", "ResidenceLength_num": "4",
    "LeisureSpend": "$100-199", "LeisureSpend_num": "3.0",
    "scenario": "demo", "Autonomy_Control": "4.0", "Platform_Comfort": "4.0",
}


def truncated(rng, lo, hi):
    """A value on [lo, hi], centred and spread like a Likert response."""
    mid = (lo + hi) / 2.0
    return round(min(hi, max(lo, rng.gauss(mid, (hi - lo) / 5.0))), 3)


def build_rows(n, seed):
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        age = rng.choice(AGE_BANDS)
        income = rng.choice(INCOME_BANDS)
        employment = rng.choice(EMPLOYMENT)
        row = {
            "PersonaID": f"SYNTHETIC-{i:04d}",
            "Age": age, "Income": income,
            "Sex": rng.choice(["Male", "Female"]),
            "Employment": employment,
            "Budget": rng.choice(["low", "medium", "high"]),
            "TopRated": rng.choice(["yes", "no"]),
            "budget_tightness": truncated(rng, 1, 7),
            "Age_num": str(AGE_BANDS.index(age) + 1),
            "Income_num": str(INCOME_BANDS.index(income) + 1),
            "Sex_num": "1",
            "CityFam_num": str(rng.randint(1, 5)),
            "start_latitude": "", "start_longitude": "",
        }
        lo, hi = PSYCHOMETRIC_SCALE
        for name in PSYCHOMETRICS:
            row[name] = truncated(rng, lo, hi)
        for item, (ilo, ihi) in DEMO_ITEMS.items():
            row[item] = truncated(rng, ilo, ihi)
        row.update(CARRIED)
        rows.append(row)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20260921)
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    rows = build_rows(args.n, args.seed)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fields = list(rows[0].keys())
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} FABRICATED personas ({len(fields)} columns) to {args.out}")
    print("These are not survey data. Results from them reproduce nothing.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

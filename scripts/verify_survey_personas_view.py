#!/usr/bin/env python3
"""Assert survey.personas (Postgres) equals data/survey_personas_2026.csv.

    python3 scripts/verify_survey_personas_view.py

The ABM has two ways to reach the survey population:

    data_handoff/algo_leisure_survey_2026.csv
        |-- data/build_survey_personas.py --> data/survey_personas_2026.csv
        |                                     (LocalDataSource, no DB needed)
        '-- db/load_survey.py -> survey.respondents -> survey.personas
                                                      (PostgresDataSource)

Both must produce the SAME 473 agent profiles, or a run on the lab server is
not the run the paper describes. The two implementations are deliberately
independent -- Python and SQL, written from the same specification -- so this
comparison is a real check rather than a tautology, in the same spirit as
scripts/verify_survey_population.py check 3 recomputing the composites.

Comparison rules, and why:
  * Numeric cells are compared as FLOATS, exactly (== on the parsed value).
    Not by string: Postgres and Python both emit shortest-round-trip decimal,
    but that is a property of two formatters agreeing, not of the data. The
    simulation parses these with float() anyway, so float equality is what
    behaviour actually depends on.
  * Text cells are compared as strings, exactly. These ARE compared literally,
    because the simulation string-matches them (Budget, TopRated, Employment,
    Age and Income band labels are looked up in params.PERSONA_MAPPING).
  * SQL NULL and the CSV's empty string are the same absence. build_survey_
    personas.py writes '' for a missing item; the view leaves it NULL; the
    datasource renders NULL back to ''.

Exits non-zero on any difference.
"""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

CSV_PATH = REPO_ROOT / "data" / "survey_personas_2026.csv"


def load_csv() -> tuple[list[str], list[dict]]:
    with CSV_PATH.open(newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        return list(rdr.fieldnames), list(rdr)


def load_view() -> tuple[list[str], list[dict]]:
    import psycopg

    dsn = os.environ.get("WELFARE_RS_PG_DSN") or os.environ.get("WELFARE_LOADER_DSN", "")
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute("SELECT * FROM survey.personas")
        cols = [d.name for d in cur.description]
        return cols, [dict(zip(cols, row)) for row in cur.fetchall()]


# ── Phase 2: the agents the two paths actually build ─────────────────────────

SKIP_ATTRS = frozenset()   # nothing is excluded; see compare_agents


def diff_value(a, b, path: str, out: list) -> None:
    """Recursive exact comparison. Floats are compared with ==, deliberately.

    Not a tolerance: both paths run identical arithmetic on identical inputs,
    so any difference at all is a real divergence, and a tolerance would hide
    exactly the class of bug this exists to catch (a scale read one way here and
    another way there). numpy scalars compare fine against Python floats.
    """
    if len(out) > 400:   # generous: a real divergence should show its full shape
        return
    if isinstance(a, dict) or isinstance(b, dict):
        if not (isinstance(a, dict) and isinstance(b, dict)) or a.keys() != b.keys():
            out.append(f"{path}: dict shape differs")
            return
        for k in a:
            diff_value(a[k], b[k], f"{path}.{k}", out)
    elif isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
        if type(a) is not type(b) or len(a) != len(b):
            out.append(f"{path}: sequence shape differs")
            return
        for i, (x, y) in enumerate(zip(a, b)):
            diff_value(x, y, f"{path}[{i}]", out)
    elif isinstance(a, float) or isinstance(b, float):
        if not (a == b or (a != a and b != b)):   # NaN == NaN treated as equal
            out.append(f"{path}: {a!r} != {b!r}")
    elif a != b:
        out.append(f"{path}: {a!r} != {b!r}")


def compare_agents(n_agents: int = 600, seed: int = 42) -> tuple[int, list]:
    """Build the population both ways and compare every agent attribute.

    Cell equality (phase 1) is necessary but not sufficient: it says the two
    sources carry the same values, not that the simulation reads them the same
    way. This builds a real population from each and compares the whole of
    vars(agent) -- psychometrics, survey items, derived behavioural
    coefficients, tastes, schedules, home and work -- so a divergence anywhere
    between the persona row and the agent shows up.

    n_agents is deliberately > 473 so _resample_persona runs too: past the
    sample size the population is extended by resampling whole profiles, and
    that draw must land on the same respondents in both paths.
    """
    import welfare_rs
    from welfare_rs import params
    from welfare_rs.datasource import PostgresDataSource
    from welfare_rs.simulation import Simulation

    rows = PostgresDataSource(os.environ["WELFARE_RS_PG_DSN"]).personas()
    if not rows:
        raise SystemExit("FATAL: PostgresDataSource.personas() returned nothing")

    net = welfare_rs.build_road_network("nyc_manhattan")
    sim_csv = Simulation(num_agents=n_agents, seed=seed, road_network=net,
                         persona_csv_path=params.SURVEY_PERSONA_CSV_PATH)
    sim_db = Simulation(num_agents=n_agents, seed=seed, road_network=net,
                        persona_rows=rows)

    if len(sim_csv.agents) != len(sim_db.agents):
        raise SystemExit(f"FATAL: agent counts differ {len(sim_csv.agents)} "
                         f"vs {len(sim_db.agents)}")
    problems: list[str] = []
    for a_csv, a_db in zip(sim_csv.agents, sim_db.agents):
        va, vb = vars(a_csv), vars(a_db)
        if va.keys() != vb.keys():
            problems.append(f"agent {a_csv.id}: attribute sets differ")
            continue
        for name in sorted(va):
            diff_value(va[name], vb[name], f"agent{a_csv.id}.{name}", problems)
        if len(problems) > 400:
            break
    return len(sim_csv.agents), problems


def main() -> int:
    csv_cols, csv_rows = load_csv()
    view_cols, view_rows = load_view()
    problems: list[str] = []

    if csv_cols != view_cols:
        only_csv = [c for c in csv_cols if c not in view_cols]
        only_view = [c for c in view_cols if c not in csv_cols]
        if only_csv or only_view:
            problems.append(f"column sets differ: only in CSV {only_csv}, "
                            f"only in view {only_view}")
        else:
            problems.append(f"column ORDER differs:\n  csv  {csv_cols}\n  view {view_cols}")
    if len(csv_rows) != len(view_rows):
        problems.append(f"row counts differ: csv {len(csv_rows)}, view {len(view_rows)}")
    if problems:
        for p in problems:
            print("FAIL: " + p)
        return 1

    csv_by_id = {r["PersonaID"]: r for r in csv_rows}
    view_by_id = {r["PersonaID"]: r for r in view_rows}
    if set(csv_by_id) != set(view_by_id):
        print(f"FAIL: PersonaID sets differ "
              f"(csv-only {sorted(set(csv_by_id) - set(view_by_id))[:5]}, "
              f"view-only {sorted(set(view_by_id) - set(csv_by_id))[:5]})")
        return 1

    n_numeric = n_text = n_null = 0
    diffs: list[str] = []
    for pid in sorted(csv_by_id):
        c_row, v_row = csv_by_id[pid], view_by_id[pid]
        for col in csv_cols:
            c_val, v_val = c_row[col], v_row[col]
            if v_val is None:
                n_null += 1
                if c_val.strip() != "":
                    diffs.append(f"{pid}.{col}: view NULL, csv {c_val!r}")
                continue
            if isinstance(v_val, str):
                n_text += 1
                if c_val != v_val:
                    diffs.append(f"{pid}.{col}: csv {c_val!r} != view {v_val!r}")
                continue
            n_numeric += 1
            if c_val.strip() == "":
                diffs.append(f"{pid}.{col}: csv empty, view {v_val!r}")
            elif float(c_val) != float(v_val):
                diffs.append(f"{pid}.{col}: csv {float(c_val)!r} != view {float(v_val)!r}")

    total = len(csv_cols) * len(csv_rows)
    if diffs:
        print(f"FAIL: {len(diffs)} of {total} cells differ. First 20:")
        for d in diffs[:20]:
            print("   " + d)
        return 1

    print("1. survey.personas == data/survey_personas_2026.csv")
    print(f"   {len(csv_rows)} rows x {len(csv_cols)} columns = {total} cells, all equal")
    print(f"   {n_numeric} numeric (float-exact), {n_text} text (literal), "
          f"{n_null} NULL/empty")

    print("\n2. both paths build identical agents")
    n_built, problems = compare_agents()
    if problems:
        by_attr: dict[str, int] = {}
        for d in problems:
            by_attr[d.split(":")[0].split(".", 1)[1]] = \
                by_attr.get(d.split(":")[0].split(".", 1)[1], 0) + 1
        print(f"   FAIL: {len(problems)} attribute differences across "
              f"{len(by_attr)} distinct attributes:")
        for attr, cnt in sorted(by_attr.items(), key=lambda kv: -kv[1]):
            print(f"      {cnt:5d}x  {attr}")
        print("   first 20 in detail:")
        for d in problems[:20]:
            print("      " + d)
        return 1
    print(f"   {n_built} agents (> 473, so whole-profile resampling ran too); "
          f"every attribute of every agent equal")
    return 0


if __name__ == "__main__":
    sys.exit(main())

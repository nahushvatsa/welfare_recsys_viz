#!/usr/bin/env python3
"""Load the Prolific survey package (data_handoff/) into the `survey` schema.

    python3 db/load_survey.py                 # verify + load everything
    python3 db/load_survey.py --verify-only   # checksums + checks, no writes
    python3 db/load_survey.py --handoff DIR

Source: the August 2026 package from Ekin Ugurel for the April 2026 survey
(IRB-FY2026-11354, NYU Tandon), N=473 after consent, completion and four
attention checks. See data_handoff/README.md.

HUMAN-SUBJECTS DATA. survey.respondents is respondent-level. Grants are set in
db/09_survey_schema.sql and deliberately do NOT include welfare_app; read the
banner there before changing them.

WHY THIS SCRIPT IS PARANOID
---------------------------
The survey is the only empirical grounding the agent population has. If it
loads subtly wrong -- a scale flipped, a level parsed as missing -- nothing
downstream fails loudly; the ABM just runs on a different population than the
paper claims. So the load asserts, and aborts on, every property the package
documents about itself:

  1. SHA-256 of all 9 files against manifest.json.
  2. 473 rows, respondent_id exactly R001..R473, no duplicates.
  3. Observed missing count == data_dictionary.n_missing, per column, all 136.
  4. Every reverse-coded _R column equals 8 - its base column (README S3).
  5. Every composite equals the unweighted mean of its construct_items.json
     items, |diff| < 1e-9 (README S6) -- checked in SQL, after the load, so it
     tests what the DB holds rather than what Python parsed.
  6. Every numeric response column lies inside its DOCUMENTED scale (README
     S3/S5), including the three band-midpoint recodes whose permitted values
     are an exact set, not a range.

THE "None" TRAP (README S8). s2_overnight_trips has a level labelled
"None (0 trips)". pandas.read_csv would silently turn it into NaN, which is how
87 responses were lost once already. This script uses the csv module, which has
no NA magic: '' is the ONLY missing marker, and it is what becomes SQL NULL.
Do not reintroduce pandas here.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys

import psycopg

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DSN = os.environ.get("WELFARE_LOADER_DSN", "")  # "" -> libpq env vars
HANDOFF = os.path.join(REPO, "data_handoff")
GENERATED_DDL = os.path.join(REPO, "db", "09b_survey_respondents.generated.sql")

SURVEY_CSV = "algo_leisure_survey_2026.csv"
N_EXPECTED = 473

# ── Column typing ────────────────────────────────────────────────────────────
#
# Storage type is decided by what a column MEANS, then verified against what it
# holds -- not read off the values alone. The distinction matters because the
# shipped file was written by pandas, which upcasts an integer column to float
# as soon as one cell is missing: 8 of the 11 plat_freq_* items and s4_search_d/e
# arrive as "3.0" purely because someone skipped them. Typing on the values
# would record that accident as a semantic difference between items on the same
# 5-point scale.
#
# So: response items are smallint; things that are genuinely means or band
# midpoints are double precision. These three dictionary blocks are the
# derived-continuous ones, which makes the rule readable rather than a list.
CONTINUOUS_BLOCKS = {
    "Latent construct (composite)",   # the 10 composites, unweighted item means
    "Derived: scenario block mean",   # s4_/s5_ *_mean
    "Derived: platform use",          # plat_any_mean, plat_search, plat_social,
                                      # plat_ai_asst
}
# Band midpoints (README S5): continuous by construction even though some
# happen to be integral in this sample.
CONTINUOUS_TYPES = {
    "numeric (years)",          # ResidenceLength_num
    "numeric (outings/week)",   # leisure_freq_num
    "numeric (trips/12 mo)",    # overnight_num
}

# ── Documented response scales (README S3, S5) ───────────────────────────────
#
# Instrument facts, not sample facts. data_dictionary.valid_values is prose for
# 108 of 136 columns, and where it does parse it sometimes gives the OBSERVED
# range (Duration_s "234-3619"), so generating bounds from it would freeze this
# sample's accidents into the schema. Columns absent here get no bound.
SCALE_RANGES = {
    # README S5, numeric recodes.
    "Age_num": (1, 6), "Sex_num": (0, 1), "Education_num": (1, 7),
    "Income_num": (1, 7), "LeisureSpend_num": (1, 5), "CityFam_num": (1, 5),
    "spur_num": (1, 5),
    # Ordinal duplicates of the 4- and 5-level habit items.
    "leisure_freq_ord": (0, 3), "overnight_ord": (0, 4),
    # Objective knowledge item, scored right/wrong.
    "s6_algo_knowledge_correct": (0, 1),
    # README S4: stated follow-through by source, 7-point.
    "s2_Q16": (1, 7), "s2_Q17": (1, 7), "s2_Q18": (1, 7),
}
# Exact permitted value sets -- stronger than a range, and available because
# README S5 enumerates the midpoints.
SCALE_SETS = {
    "ResidenceLength_num": {0.25, 0.75, 2.0, 4.0, 8.0},
    "leisure_freq_num": {0.5, 1.5, 3.5, 6.0},
    "overnight_num": {0.0, 1.5, 4.0, 8.0, 12.0},
}


def scale_of(name: str) -> tuple | None:
    """Documented (lo, hi) for a numeric column, or None for unbounded."""
    if name in SCALE_RANGES:
        return SCALE_RANGES[name]
    # README S3: "All platform-frequency items are 5-point (1 = Never ...
    # 5 = Always)", and the four aggregates are means of those items.
    if name.startswith("plat_freq_") or name.startswith("plat_"):
        return (1, 5)
    # README S3: "All Likert items are 7-point." Every s3_/s4_/s5_/s6_/s7_
    # numeric column is a Likert item or a mean of them; the one exception is
    # the knowledge item, pinned above.
    if name.startswith(("s3_", "s4_", "s5_", "s6_", "s7_")):
        return (1, 7)
    return None


def sql_type(name: str, block: str, data_type: str, values: list[str]) -> str:
    """Storage type for one column: meaning first, then verified on the data."""
    if name == "respondent_id":
        return "text"
    nonempty = [v for v in values if v.strip() != ""]
    if not nonempty:
        return "text"
    if set(nonempty) <= {"True", "False"}:
        return "boolean"
    try:
        nums = [float(v) for v in nonempty]
    except ValueError:
        return "text"
    if block in CONTINUOUS_BLOCKS or data_type in CONTINUOUS_TYPES:
        return "double precision"
    if all(n == int(n) for n in nums):
        return "smallint" if all(-32768 <= n <= 32767 for n in nums) else "integer"
    # A column we classified as a response item but which holds fractions:
    # the meaning-first rule and the data disagree. Do not paper over it.
    raise SystemExit(
        f"FATAL: {name!r} (block={block!r}, type={data_type!r}) is not a derived "
        f"continuous column but holds non-integral values. Reclassify it "
        f"deliberately rather than letting the loader guess."
    )


def cast(raw: str, pgtype: str):
    """CSV cell -> Python value. '' is the only missing marker (see module doc)."""
    if raw.strip() == "":
        return None
    if pgtype == "boolean":
        return raw == "True"
    if pgtype in ("smallint", "integer"):
        return int(float(raw))
    if pgtype == "double precision":
        return float(raw)
    return raw


def quote_ident(name: str) -> str:
    if not name.replace("_", "").isalnum():
        raise SystemExit(f"FATAL: unexpected character in column name {name!r}")
    return f'"{name}"'


# ── Package verification ─────────────────────────────────────────────────────

def verify_manifest(handoff: str) -> list[dict]:
    """SHA-256 every shipped file against manifest.json. Aborts on mismatch."""
    with open(os.path.join(handoff, "manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    if manifest.get("n_respondents") != N_EXPECTED:
        raise SystemExit(f"FATAL: manifest says n_respondents="
                         f"{manifest.get('n_respondents')}, expected {N_EXPECTED}")
    print(f"  manifest: {manifest['package']}")
    print(f"            {manifest['study']}, built {manifest['built']}")
    for entry in manifest["files"]:
        path = os.path.join(handoff, entry["name"])
        if not os.path.exists(path):
            raise SystemExit(f"FATAL: manifest lists {entry['name']}, not on disk")
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        got, want = h.hexdigest(), entry["sha256"]
        size = os.path.getsize(path)
        if got != want:
            raise SystemExit(f"FATAL: {entry['name']} sha256 {got[:16]}... != "
                             f"manifest {want[:16]}...  THE FILE IS NOT WHAT WAS SHIPPED")
        if size != entry["bytes"]:
            raise SystemExit(f"FATAL: {entry['name']} is {size} bytes, "
                             f"manifest says {entry['bytes']}")
        print(f"    ok  {entry['name']:38s} {size:>9,} B  {got[:12]}...")
    return manifest["files"]


def check_respondents(rows: list[dict], dd: dict) -> None:
    """Every property the package documents about the respondent file."""
    if len(rows) != N_EXPECTED:
        raise SystemExit(f"FATAL: {len(rows)} rows, expected {N_EXPECTED}")
    ids = [r["respondent_id"] for r in rows]
    expected_ids = [f"R{i:03d}" for i in range(1, N_EXPECTED + 1)]
    if ids != expected_ids:
        dupes = len(ids) - len(set(ids))
        raise SystemExit(f"FATAL: respondent_id is not exactly R001..R{N_EXPECTED} "
                         f"in order ({dupes} duplicates)")
    cols = list(rows[0].keys())
    if set(cols) != set(dd):
        missing = set(dd) - set(cols)
        extra = set(cols) - set(dd)
        raise SystemExit(f"FATAL: dictionary/file column mismatch. "
                         f"only in dictionary: {sorted(missing)}; "
                         f"only in file: {sorted(extra)}")
    print(f"  {len(rows)} rows x {len(cols)} columns; ids R001..R{N_EXPECTED}; "
          f"dictionary covers every column")

    # Declared vs observed missingness, per column.
    bad = []
    for c in cols:
        observed = sum(1 for r in rows if r[c].strip() == "")
        declared = int(dd[c]["n_missing"])
        if observed != declared:
            bad.append(f"{c}: observed {observed}, dictionary says {declared}")
    if bad:
        raise SystemExit("FATAL: missingness disagrees with the dictionary:\n    "
                         + "\n    ".join(bad))
    total = sum(int(dd[c]["n_missing"]) for c in cols)
    print(f"  missingness matches the dictionary for all {len(cols)} columns "
          f"({total} missing cells)")

    # Reverse coding: _R == 8 - base (README S3).
    rev = [c for c in cols if c.endswith("_R")]
    for c in rev:
        base = c[:-2]
        if base not in cols:
            raise SystemExit(f"FATAL: {c} has no base column {base}")
        for r in rows:
            if r[base].strip() == "" or r[c].strip() == "":
                continue
            if abs((8 - float(r[base])) - float(r[c])) > 1e-12:
                raise SystemExit(f"FATAL: {r['respondent_id']} {c}={r[c]} but "
                                 f"8-{base}={8 - float(r[base])}")
    print(f"  all {len(rev)} reverse-coded columns satisfy _R = 8 - base exactly")

    # Documented scales (README S3, S5).
    bounded = 0
    for c in cols:
        vals = [float(r[c]) for r in rows
                if r[c].strip() != "" and _is_num(r[c])]
        if not vals:
            continue
        if c in SCALE_SETS:
            off = sorted({v for v in vals} - SCALE_SETS[c])
            if off:
                raise SystemExit(f"FATAL: {c} holds {off}, outside the documented "
                                 f"midpoints {sorted(SCALE_SETS[c])}")
            bounded += 1
            continue
        rng = scale_of(c)
        if rng is None:
            continue
        lo, hi = rng
        off = [v for v in vals if not (lo <= v <= hi)]
        if off:
            raise SystemExit(f"FATAL: {c} holds values outside its documented "
                             f"{lo}-{hi} scale: {sorted(set(off))[:8]}")
        bounded += 1
    print(f"  {bounded} numeric columns verified inside their documented scales")


def _is_num(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


# ── Load phases ──────────────────────────────────────────────────────────────

def read_csv_rows(path: str) -> list[dict]:
    """csv module only -- see the "None" trap in the module docstring."""
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_ddl(cols: list[str], types: dict[str, str]) -> str:
    """Generate survey.respondents. Written to disk for review, not hidden."""
    width = max(len(quote_ident(c)) for c in cols)
    lines = [
        "-- GENERATED by db/load_survey.py -- do not edit by hand.",
        "-- 136 typed columns derived from survey.data_dictionary and verified",
        "-- against the shipped file. Column meanings, item wording and valid",
        "-- values live in survey.data_dictionary, joined on `variable`; they are",
        "-- deliberately not duplicated here.",
        "",
        "CREATE TABLE IF NOT EXISTS survey.respondents (",
    ]
    for c in cols:
        ident = quote_ident(c)
        nn = " NOT NULL" if c == "respondent_id" else ""
        lines.append(f"    {ident:<{width}} {types[c]}{nn},")
    lines.append("    PRIMARY KEY (respondent_id),")
    lines.append("    CONSTRAINT respondents_id_format "
                 "CHECK (respondent_id ~ '^R[0-9]{3}$')")
    lines.append(");")
    lines.append("")
    lines.append("COMMENT ON TABLE survey.respondents IS")
    lines.append("    'Respondent-level survey data, IRB-FY2026-11354, N=473. "
                 "HUMAN SUBJECTS: '")
    lines.append("    'not readable by welfare_app; the ABM reads survey.personas. "
                 "Column '")
    lines.append("    'documentation is in survey.data_dictionary.';")
    return "\n".join(lines) + "\n"


def load_respondents(cur, handoff: str, dd: dict) -> list[dict]:
    rows = read_csv_rows(os.path.join(handoff, SURVEY_CSV))
    check_respondents(rows, dd)

    cols = list(rows[0].keys())
    types = {c: sql_type(c, dd[c]["block"], dd[c]["data_type"],
                         [r[c] for r in rows]) for c in cols}
    ddl = build_ddl(cols, types)
    with open(GENERATED_DDL, "w", encoding="utf-8") as f:
        f.write(ddl)
    print(f"  generated DDL -> {os.path.relpath(GENERATED_DDL, REPO)}")
    counts = {}
    for t in types.values():
        counts[t] = counts.get(t, 0) + 1
    print("  column types: " + ", ".join(f"{v} {k}" for k, v in sorted(counts.items())))

    _ensure_table(cur, ddl, cols, types)

    idents = ", ".join(quote_ident(c) for c in cols)
    with cur.copy(f"COPY survey.respondents ({idents}) FROM STDIN") as cp:
        for r in rows:
            cp.write_row(tuple(cast(r[c], types[c]) for c in cols))
    cur.execute("SELECT count(*) FROM survey.respondents")
    print(f"  survey.respondents: {cur.fetchone()[0]} rows")
    return rows


def _ensure_table(cur, ddl: str, cols: list[str], types: dict[str, str]) -> None:
    """Create survey.respondents, or reload it in place if it already matches.

    NOT ``DROP TABLE ... CASCADE``. survey.personas depends on this table, and
    cascading would drop the view, its GRANT to welfare_app and its comment --
    after which the app keeps running, silently falling back to the persona CSV
    (PostgresDataSource.personas() warns, but a warning in a worker log is not a
    stop). A reload must never be able to change which population the server
    serves.

    So: same columns and types -> TRUNCATE and refill, leaving the view and its
    privileges untouched. Different -> stop and make a human do it, because a
    generated schema that has changed shape is exactly the thing that should not
    be applied by a script that was only asked to reload data.
    """
    cur.execute("""
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_schema = 'survey' AND table_name = 'respondents'
        ORDER BY ordinal_position
    """)
    existing = cur.fetchall()
    if not existing:
        cur.execute(ddl)
        print("  survey.respondents: created")
        return

    want = [(c, types[c]) for c in cols]
    if [(r[0], r[1]) for r in existing] == want:
        cur.execute("TRUNCATE survey.respondents")
        print("  survey.respondents: schema unchanged, truncated for reload")
        return

    got = {r[0]: r[1] for r in existing}
    diffs = [f"    {c}: table has {got.get(c, '<missing>')}, generated {t}"
             for c, t in want if got.get(c) != t]
    diffs += [f"    {c}: in table, not generated" for c in got if c not in types]
    raise SystemExit(
        "FATAL: survey.respondents exists with a DIFFERENT schema than the one "
        "generated from the package:\n" + "\n".join(diffs[:20]) + "\n\n"
        "  Not applied automatically: survey.personas depends on this table, so "
        "replacing it means dropping and rebuilding the view and its grant.\n"
        "  To apply deliberately:\n"
        "    psql -c 'DROP VIEW survey.personas; DROP TABLE survey.respondents;'\n"
        "    python3 db/load_survey.py\n"
        "    psql -v ON_ERROR_STOP=1 -f db/10_survey_personas.sql\n"
        "    python3 scripts/verify_survey_personas_view.py"
    )


def load_dictionary(cur, handoff: str) -> dict:
    rows = read_csv_rows(os.path.join(handoff, "data_dictionary.csv"))
    cur.execute("TRUNCATE survey.data_dictionary CASCADE")
    with cur.copy(
        "COPY survey.data_dictionary (variable, block, source_question, data_type,"
        " valid_values, n_missing, missing_convention, latent_construct,"
        " reverse_coded, description) FROM STDIN"
    ) as cp:
        for r in rows:
            cp.write_row((r["variable"], r["block"], r["source_question"],
                          r["data_type"], r["valid_values"], int(r["n_missing"]),
                          r["missing_convention"], r["latent_construct"],
                          r["reverse_coded"], r["description"]))
    print(f"  survey.data_dictionary: {len(rows)} rows")
    return {r["variable"]: r for r in rows}


def load_constructs(cur, handoff: str) -> dict:
    with open(os.path.join(handoff, "construct_items.json"), encoding="utf-8") as f:
        lv = json.load(f)
    cur.execute("TRUNCATE survey.construct_items")
    with cur.copy("COPY survey.construct_items (construct, item, item_ord) "
                  "FROM STDIN") as cp:
        for name, items in lv.items():
            for i, item in enumerate(items):
                cp.write_row((name, item, i))
    print(f"  survey.construct_items: {len(lv)} constructs, "
          f"{sum(len(v) for v in lv.values())} items")
    return lv


def load_marginals(cur, handoff: str) -> None:
    rows = read_csv_rows(os.path.join(handoff, "calibration_marginals.csv"))
    cur.execute("TRUNCATE survey.marginals")
    with cur.copy("COPY survey.marginals (variable, kind, level, statistic,"
                  " value, proportion) FROM STDIN") as cp:
        for r in rows:
            cp.write_row((
                r["variable"], r["kind"], r["level"] or None, r["statistic"] or None,
                float(r["value"]) if r["value"].strip() else None,
                float(r["proportion"]) if r["proportion"].strip() else None,
            ))
    print(f"  survey.marginals: {len(rows)} rows")


def _read_matrix(path: str) -> tuple[list[str], dict]:
    """Square labelled matrix CSV -> (labels, {(a,b): value})."""
    with open(path, newline="", encoding="utf-8") as f:
        rdr = list(csv.reader(f))
    labels = rdr[0][1:]
    cells = {}
    for row in rdr[1:]:
        a = row[0]
        for b, v in zip(labels, row[1:]):
            cells[(a, b)] = v
    if sorted({a for a, _ in cells}) != sorted(labels):
        raise SystemExit(f"FATAL: {os.path.basename(path)} is not square")
    return labels, cells


def load_correlations(cur, handoff: str) -> None:
    labels, r_cells = _read_matrix(os.path.join(handoff, "calibration_correlations.csv"))
    n_labels, n_cells = _read_matrix(os.path.join(handoff, "calibration_correlations_n.csv"))
    if labels != n_labels:
        raise SystemExit("FATAL: correlation and pairwise-N matrices label differently")
    with open(os.path.join(handoff, "calibration_correlations_meta.json"),
              encoding="utf-8") as f:
        meta = json.load(f)
    if meta["attributes"] != labels:
        raise SystemExit("FATAL: meta.attributes does not match the matrix labels")
    if not meta.get("positive_semidefinite"):
        raise SystemExit("FATAL: shipped correlation matrix is not PSD")

    # Symmetry and unit diagonal, checked before storing rather than assumed.
    for a in labels:
        if abs(float(r_cells[(a, a)]) - 1.0) > 1e-12:
            raise SystemExit(f"FATAL: diagonal r[{a},{a}] != 1")
        for b in labels:
            if abs(float(r_cells[(a, b)]) - float(r_cells[(b, a)])) > 1e-12:
                raise SystemExit(f"FATAL: correlation matrix asymmetric at ({a},{b})")

    cur.execute("TRUNCATE survey.correlations")
    with cur.copy("COPY survey.correlations (var_a, var_b, pearson_r, pairwise_n)"
                  " FROM STDIN") as cp:
        for a in labels:
            for b in labels:
                cp.write_row((a, b, float(r_cells[(a, b)]), int(float(n_cells[(a, b)]))))
    cur.execute("DELETE FROM survey.calibration_meta")
    cur.execute("INSERT INTO survey.calibration_meta (meta) VALUES (%s)",
                (json.dumps(meta),))
    print(f"  survey.correlations: {len(labels)}x{len(labels)} = "
          f"{len(labels) ** 2} pairs over {len(labels)} agent attributes "
          f"(PSD, min eigenvalue {meta['min_eigenvalue']:.4f}, "
          f"n_complete={meta['n_complete_cases']})")


def load_package_files(cur, files: list[dict]) -> None:
    cur.execute("TRUNCATE survey.package_files")
    with cur.copy("COPY survey.package_files (name, bytes, sha256, description)"
                  " FROM STDIN") as cp:
        for e in files:
            cp.write_row((e["name"], e["bytes"], e["sha256"], e.get("description")))
    print(f"  survey.package_files: {len(files)} files")


# ── Post-load validation, in SQL ─────────────────────────────────────────────

def verify_in_db(cur) -> None:
    """Re-check the package's own invariants against what the DB now holds.

    Deliberately reads construct membership back OUT of survey.construct_items
    rather than from the JSON still in memory, and recomputes the composites in
    SQL. A Python-side check would only prove Python parsed the CSV correctly;
    this proves the DB holds the same survey, through the COPY and the type
    assignment.
    """
    cur.execute("SELECT count(*) FROM survey.respondents")
    n = cur.fetchone()[0]
    if n != N_EXPECTED:
        raise SystemExit(f"FATAL: survey.respondents holds {n} rows, expected {N_EXPECTED}")

    cur.execute("SELECT construct, array_agg(item ORDER BY item_ord) "
                "FROM survey.construct_items GROUP BY construct ORDER BY construct")
    constructs = cur.fetchall()
    worst = 0.0
    for name, items in constructs:
        # Explicit ::double precision on the first term: the items are smallint,
        # and smallint arithmetic in Postgres is INTEGER division, which would
        # silently make every composite check pass against a wrong mean.
        terms = " + ".join(
            (f"{quote_ident(i)}::double precision" if k == 0 else quote_ident(i))
            for k, i in enumerate(items)
        )
        cur.execute(
            f"SELECT max(abs(({terms}) / {len(items)} - {quote_ident(name)})) "
            f"FROM survey.respondents"
        )
        diff = cur.fetchone()[0]
        if diff is None or diff > 1e-9:
            raise SystemExit(f"FATAL: composite {name} != mean of its "
                             f"{len(items)} items in the DB (max diff {diff})")
        worst = max(worst, float(diff))
    print(f"  all {len(constructs)} composites reproduce from their items in SQL "
          f"(worst |diff| {worst:.2e}, tolerance 1e-9)")

    # Missingness, recounted from the loaded table against the loaded dictionary.
    cur.execute("SELECT variable, n_missing FROM survey.data_dictionary ORDER BY variable")
    declared = dict(cur.fetchall())
    checks = " UNION ALL ".join(
        f"SELECT {v!r} AS variable, count(*) - count({quote_ident(v)}) AS n_null "
        f"FROM survey.respondents" for v in declared
    )
    cur.execute(f"SELECT variable, n_null FROM ({checks}) t ORDER BY variable")
    bad = [(v, n_null, declared[v]) for v, n_null in cur.fetchall()
           if n_null != declared[v]]
    if bad:
        raise SystemExit("FATAL: NULL counts in the DB disagree with the "
                         f"dictionary: {bad[:10]}")
    print(f"  NULL counts in the DB match the dictionary for all "
          f"{len(declared)} columns")

    cur.execute("SELECT count(*) FROM survey.correlations WHERE pearson_r IS NULL")
    if cur.fetchone()[0]:
        raise SystemExit("FATAL: null correlations loaded")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--handoff", default=HANDOFF, help="package directory")
    ap.add_argument("--verify-only", action="store_true",
                    help="checksum and check the package; touch no tables")
    args = ap.parse_args()

    print("== package provenance ==")
    files = verify_manifest(args.handoff)

    if args.verify_only:
        dd = {r["variable"]: r for r in
              read_csv_rows(os.path.join(args.handoff, "data_dictionary.csv"))}
        print("\n== respondent file ==")
        check_respondents(read_csv_rows(os.path.join(args.handoff, SURVEY_CSV)), dd)
        print("\nverify-only: no tables were written.")
        return 0

    with psycopg.connect(DSN) as conn:
        with conn.cursor() as cur:
            print("\n== load ==")
            dd = load_dictionary(cur, args.handoff)
            load_respondents(cur, args.handoff, dd)
            load_constructs(cur, args.handoff)
            load_marginals(cur, args.handoff)
            load_correlations(cur, args.handoff)
            load_package_files(cur, files)
            print("\n== verification against the loaded tables ==")
            verify_in_db(cur)
        conn.commit()
    print("\nsurvey package loaded and verified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

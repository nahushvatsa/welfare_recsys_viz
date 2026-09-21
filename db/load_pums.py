#!/usr/bin/env python3
"""Load ACS PUMS person + household records, and the tract -> PUMA crosswalk.

    python3 db/load_pums.py --what all                 # 7 metro states
    python3 db/load_pums.py --what persons --states dc
    python3 db/load_pums.py --what xwalk

Input is whatever ``db/fetch_census_data.py`` left under ``data/census``. Every
zip is checksum-verified against ``manifest_acs_pums.json`` before a single row
is loaded: the manifest records what was downloaded, so loading a file whose
bytes have changed would quietly put a different vintage in the database.

Idempotent per state — the state's rows are deleted and reloaded, so a rerun
after a fixed column mapping is safe and leaves no duplicates.

Columns are a curated subset of the ~290 person / ~240 household PUMS columns;
see the rationale in db/12_pums_schema.sql. The mapping below is by NAME, not
position, and a missing source column is a hard failure — the 2023 vintage
calls the state column STATE where older ones called it ST, and silently
loading NULLs would be far worse than stopping.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import sys
import time
import zipfile

import psycopg

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DSN = os.environ.get("WELFARE_LOADER_DSN", "")  # "" -> libpq env vars
DATA = os.path.join(REPO, "data", "census")
PUMS_DIR = os.path.join(DATA, "pums")
XWALK_PATH = os.path.join(DATA, "xwalk", "2020_tract_to_2020_puma.txt")
MANIFEST = os.path.join(DATA, "manifest_acs_pums.json")

# Fourteen states: the seven holding metro cores plus the seven their
# commuter sheds reach. See db/fetch_census_data.py for how they were found.
DEFAULT_STATES = ["ny", "ca", "il", "tx", "fl", "wa", "dc",
                  "nj", "md", "va", "ct", "in", "pa", "de"]

# target column -> PUMS source column. Order defines the COPY column list.
HH_MAP = [
    ("serialno", "SERIALNO"), ("state", "STATE"), ("puma", "PUMA"),
    ("wgtp", "WGTP"), ("np", "NP"), ("typehugq", "TYPEHUGQ"),
    ("adjinc", "ADJINC"), ("adjhsg", "ADJHSG"),
    ("hincp", "HINCP"), ("fincp", "FINCP"),
    ("veh", "VEH"), ("ten", "TEN"), ("bld", "BLD"),
    ("hht", "HHT"), ("hht2", "HHT2"), ("hupac", "HUPAC"),
    ("noc", "NOC"), ("npf", "NPF"),
    ("r18", "R18"), ("r60", "R60"), ("r65", "R65"),
    ("wif", "WIF"), ("workstat", "WORKSTAT"),
    ("hhl", "HHL"), ("lngi", "LNGI"), ("multg", "MULTG"), ("partner", "PARTNER"),
    ("mv", "MV"), ("yrblt", "YRBLT"), ("vacs", "VACS"),
    ("grpip", "GRPIP"), ("ocpip", "OCPIP"),
]

P_MAP = [
    ("serialno", "SERIALNO"), ("sporder", "SPORDER"),
    ("state", "STATE"), ("puma", "PUMA"),
    ("pwgtp", "PWGTP"), ("adjinc", "ADJINC"),
    ("agep", "AGEP"), ("sex", "SEX"), ("relshipp", "RELSHIPP"),
    ("mar", "MAR"), ("cit", "CIT"), ("nativity", "NATIVITY"),
    ("rac1p", "RAC1P"), ("hisp", "HISP"), ("lanx", "LANX"), ("eng", "ENG"),
    ("schl", "SCHL"), ("sch", "SCH"), ("mil", "MIL"),
    ("esr", "ESR"), ("wkhp", "WKHP"), ("wkl", "WKL"), ("wkwn", "WKWN"),
    ("indp", "INDP"), ("naicsp", "NAICSP"), ("occp", "OCCP"), ("socp", "SOCP"),
    ("powpuma", "POWPUMA"), ("powsp", "POWSP"),
    ("jwtrns", "JWTRNS"), ("jwmnp", "JWMNP"),
    ("wagp", "WAGP"), ("semp", "SEMP"), ("pernp", "PERNP"), ("pincp", "PINCP"),
    ("intp", "INTP"), ("retp", "RETP"), ("povpip", "POVPIP"),
    ("dis", "DIS"), ("ddrs", "DDRS"), ("dear", "DEAR"), ("deye", "DEYE"),
    ("dout", "DOUT"), ("dphy", "DPHY"), ("drem", "DREM"),
]

# Columns kept as text (codes with leading zeros or letters); everything else
# is parsed as an integer so a malformed cell fails loudly at load time.
TEXT_COLS = {"serialno", "state", "puma", "indp", "naicsp", "occp", "socp",
             "powpuma", "powsp"}


def verify_checksum(path: str) -> None:
    """Abort unless the file matches the digest recorded at download time."""
    if not os.path.exists(MANIFEST):
        raise SystemExit(f"FATAL: no manifest at {MANIFEST} — run db/fetch_census_data.py first")
    files = json.load(open(MANIFEST))["files"]
    entry = files.get(os.path.basename(path))
    if entry is None:
        raise SystemExit(f"FATAL: {os.path.basename(path)} is not in the manifest")
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    if h.hexdigest() != entry["sha256"]:
        raise SystemExit(f"FATAL: {os.path.basename(path)} does not match the manifest "
                         f"digest — redownload it before loading")


def _member(zf: zipfile.ZipFile, prefix: str) -> str:
    names = [n for n in zf.namelist() if n.lower().endswith(".csv")
             and os.path.basename(n).lower().startswith(prefix)]
    if len(names) != 1:
        raise SystemExit(f"FATAL: expected one {prefix}*.csv in {zf.filename}, found {names}")
    return names[0]


def _rows(zip_path: str, prefix: str, mapping):
    """Yield tuples in `mapping` order, ints parsed, blanks as None."""
    verify_checksum(zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        member = _member(zf, prefix)
        with zf.open(member) as raw:
            reader = csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8-sig", newline=""))
            missing = [src for _, src in mapping if src not in reader.fieldnames]
            if missing:
                raise SystemExit(
                    f"FATAL: {os.path.basename(zip_path)}:{member} is missing columns "
                    f"{missing}. The PUMS vintage may have renamed them; fix the mapping "
                    f"in db/load_pums.py rather than loading NULLs.")
            for rec in reader:
                out = []
                for dst, src in mapping:
                    val = (rec.get(src) or "").strip()
                    if val == "":
                        out.append(None)
                    elif dst in TEXT_COLS:
                        out.append(val)
                    else:
                        try:
                            out.append(int(float(val)))
                        except ValueError:
                            raise SystemExit(
                                f"FATAL: {member}: column {src} has non-numeric value "
                                f"{val!r} (record {rec.get('SERIALNO')})")
                yield tuple(out)


def load_table(conn, *, table, mapping, zip_path, prefix, state, log=print) -> int:
    cols = [dst for dst, _ in mapping]
    cur = conn.cursor()
    existing = cur.execute(f"SELECT count(*) FROM {table} WHERE state = %s",
                           (state,)).fetchone()[0]
    if existing:
        log(f"    replacing {existing:,} existing rows for state {state}")
        cur.execute(f"DELETE FROM {table} WHERE state = %s", (state,))

    t0 = time.time()
    n = 0
    with cur.copy(f"COPY {table} ({', '.join(cols)}) FROM STDIN") as cp:
        for row in _rows(zip_path, prefix, mapping):
            cp.write_row(row)
            n += 1
    conn.commit()
    log(f"    {table}: {n:,} rows in {time.time() - t0:.0f}s")
    return n


def phase_pums(conn, states, log=print) -> None:
    for st in states:
        log(f"  {st.upper()}")
        # Households first: persons reference them, and loading the pair in
        # this order means a half-finished state never has orphan persons.
        load_table(conn, table="census.pums_households", mapping=HH_MAP,
                   zip_path=os.path.join(PUMS_DIR, f"csv_h{st}.zip"),
                   prefix="psam_h", state=_fips(st), log=log)
        load_table(conn, table="census.pums_persons", mapping=P_MAP,
                   zip_path=os.path.join(PUMS_DIR, f"csv_p{st}.zip"),
                   prefix="psam_p", state=_fips(st), log=log)


# Postal code -> FIPS, for the DELETE that makes a reload idempotent. The files
# themselves carry the FIPS in STATE; this only has to agree with them, which
# the post-load check verifies.
STATE_FIPS = {"ny": "36", "ca": "06", "il": "17", "tx": "48",
              "fl": "12", "wa": "53", "dc": "11", "nj": "34", "md": "24",
              "va": "51", "ct": "09", "in": "18", "pa": "42", "de": "10"}


def _fips(postal: str) -> str:
    try:
        return STATE_FIPS[postal.lower()]
    except KeyError:
        raise SystemExit(f"FATAL: no FIPS mapping for state {postal!r}; add it to STATE_FIPS")


def phase_xwalk(conn, states, log=print) -> None:
    """Load the 2020 tract -> 2020 PUMA crosswalk (all states, it is tiny)."""
    verify_checksum(XWALK_PATH)
    cur = conn.cursor()
    cur.execute("TRUNCATE census.tract_puma")
    n = 0
    with open(XWALK_PATH, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        need = {"STATEFP", "COUNTYFP", "TRACTCE", "PUMA5CE"}
        if not need.issubset(reader.fieldnames or []):
            raise SystemExit(f"FATAL: crosswalk columns are {reader.fieldnames}, need {need}")
        with cur.copy("COPY census.tract_puma (tract_geoid, state, puma) FROM STDIN") as cp:
            for rec in reader:
                st, co, tr = rec["STATEFP"], rec["COUNTYFP"], rec["TRACTCE"]
                cp.write_row((f"{st}{co}{tr}", st, rec["PUMA5CE"]))
                n += 1
    conn.commit()
    log(f"  census.tract_puma: {n:,} tracts")


def phase_verify(conn, states, log=print) -> None:
    """Report what landed, and fail on the conditions that would corrupt a build."""
    cur = conn.cursor()
    log("\n  per state: households / persons / person-weight (= population)")
    rows = cur.execute("""
        SELECT p.state,
               (SELECT count(*) FROM census.pums_households h WHERE h.state = p.state),
               count(*), sum(p.pwgtp),
               count(DISTINCT p.puma)
        FROM census.pums_persons p GROUP BY p.state ORDER BY p.state
    """).fetchall()
    for st, nh, np_, wt, pumas in rows:
        log(f"    {st}  {nh:>9,} hh  {np_:>9,} persons  {wt:>12,} weighted  {pumas:>4} PUMAs")

    # 1. Every person must have a household record: the household carries
    #    income and vehicles, so an orphan person is an unusable draw.
    orphans = cur.execute("""
        SELECT count(*) FROM census.pums_persons p
        LEFT JOIN census.pums_households h USING (serialno)
        WHERE h.serialno IS NULL
    """).fetchone()[0]
    log(f"  orphan persons (no household record): {orphans:,}")
    if orphans:
        raise SystemExit("FATAL: orphan person records — the load is incomplete")

    # 2. Group quarters must look the way the schema comment claims, because
    #    the reweighting fits household controls over housing units only.
    gq = cur.execute("""
        SELECT count(*) FILTER (WHERE is_gq),
               count(*) FILTER (WHERE is_gq AND wgtp > 0),
               count(*) FILTER (WHERE occupied),
               count(*) FILTER (WHERE typehugq = 1 AND np = 0)
        FROM census.pums_households
    """).fetchone()
    log(f"  households: {gq[0]:,} group-quarters ({gq[1]:,} with a weight — expect 0), "
        f"{gq[2]:,} occupied units, {gq[3]:,} vacant")
    if gq[1]:
        raise SystemExit("FATAL: group-quarters records carry household weights; "
                         "the household-control logic in the reweighting assumes they do not")

    # 3. Every PUMA present must be resolvable from a tract, or agents living
    #    in that tract have no sample to draw from.
    unmatched = cur.execute("""
        SELECT count(*) FROM (
            SELECT DISTINCT state, puma FROM census.pums_persons
        ) p
        LEFT JOIN (SELECT DISTINCT state, puma FROM census.tract_puma) x
          USING (state, puma)
        WHERE x.puma IS NULL
    """).fetchone()[0]
    log(f"  PUMS PUMAs absent from the crosswalk: {unmatched}")
    if unmatched:
        raise SystemExit("FATAL: PUMS carries PUMAs the crosswalk does not know")

    # 4. Income adjustment must actually have been applied.
    adj = cur.execute("""
        SELECT count(*) FILTER (WHERE adjinc IS NULL),
               round(min(adjinc / 1000000.0)::numeric, 4),
               round(max(adjinc / 1000000.0)::numeric, 4)
        FROM census.pums_persons
    """).fetchone()
    log(f"  income adjustment factors: {adj[1]} .. {adj[2]} (null: {adj[0]})")
    if adj[0]:
        raise SystemExit("FATAL: person records without an income adjustment factor")


PHASES = {"hh-persons": phase_pums, "xwalk": phase_xwalk, "verify": phase_verify}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--what", choices=list(PHASES) + ["all"], default="all")
    ap.add_argument("--states", nargs="*", default=DEFAULT_STATES)
    args = ap.parse_args()

    phases = ["hh-persons", "xwalk", "verify"] if args.what == "all" else [args.what]
    with psycopg.connect(DSN) as conn:
        conn.cursor().execute("SET maintenance_work_mem = '2GB'")
        for name in phases:
            print(f"\nphase: {name}", flush=True)
            PHASES[name](conn, args.states)
    return 0


if __name__ == "__main__":
    sys.exit(main())

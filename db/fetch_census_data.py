#!/usr/bin/env python3
"""Download the Census inputs for the ACS/PUMS population build.

Three sources, all public and free of any key or licence:

* **ACS 5-year detailed tables**, table-based summary file. One ``.dat`` per
  table, pipe-delimited, all geographies in one file (US, state, county, tract
  and — for tables published there — block group). These supply the small-area
  control totals the PUMS reweighting fits to.
* **ACS PUMS 5-year**, person and household files per state. These are the
  person-level joint distribution the population is drawn from.
* **2020 tract -> PUMA crosswalk**, which links the two: a LODES home block
  gives a tract, the tract gives the PUMA whose PUMS sample describes it.

Every file is checksummed into ``manifest_acs_pums.json`` next to the data, so
a rebuild can prove it used the same bytes. Re-running skips files already
present whose SHA-256 matches the manifest; ``--force`` redownloads.

Vintage is pinned by ``--year`` (default 2023, matching LODES8 2023). ACS
5-year and LODES both sit on 2020 census geography, so the two agree without a
crosswalk of their own.

Run:  python db/fetch_census_data.py --what all
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "data" / "census"

# Every state an agent HOME can fall in. Seven hold the metro cores
# (params.METRO_PARAMS); the other seven are reached by the commuter sheds and
# were found by grouping public.metro_od_pairs on left(h_geoid, 2): New Jersey
# alone holds 3.5% of all jobs, Maryland 2.7%, Virginia 1.7%. PUMS is needed
# wherever an agent LIVES, not merely where it works, so all fourteen load.
PUMS_STATES = ["ny", "ca", "il", "tx", "fl", "wa", "dc",
               "nj", "md", "va", "ct", "in", "pa", "de"]

# ACS tables, and what each one controls in the reweighting.
ACS_TABLES = {
    "b01001": "sex by age",
    "b19001": "household income (16 bands)",
    "b23025": "employment status, population 16+",
    "b25044": "tenure by vehicles available",
    "b08301": "means of transportation to work",
    "b11016": "household type by household size",
    "b18101": "sex by age by disability status (tract only)",
    "b08201": "household size by vehicles available (tract only)",
}

ACS_BASE = "https://www2.census.gov/programs-surveys/acs/summary_file/{year}/table-based-SF/data/5YRData"
PUMS_BASE = "https://www2.census.gov/programs-surveys/acs/data/pums/{year}/5-Year"
XWALK_URL = ("https://www2.census.gov/geo/docs/maps-data/data/rel2020/"
             "2020_Census_Tract_to_2020_PUMA.txt")

CHUNK = 1 << 20
RETRIES = 4


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def download(url: str, dest: Path, *, force: bool, manifest: dict, log=print) -> dict:
    """Fetch one file, verify it, and return its manifest entry.

    Skips the download when the file is already on disk and its digest matches
    the manifest — the manifest is the authority, so a truncated or edited file
    is redownloaded rather than trusted.
    """
    key = dest.name
    prior = manifest.get(key)
    if dest.exists() and not force and prior:
        if sha256_of(dest) == prior.get("sha256"):
            log(f"  skip   {key} (verified {prior['bytes']:,} bytes)")
            return prior
        log(f"  stale  {key} — digest differs from manifest, redownloading")

    tmp = dest.with_suffix(dest.suffix + ".part")
    last_err = None
    for attempt in range(1, RETRIES + 1):
        try:
            t0 = time.time()
            with urllib.request.urlopen(url, timeout=120) as resp, tmp.open("wb") as out:
                declared = resp.headers.get("Content-Length")
                declared = int(declared) if declared else None
                got = 0
                while True:
                    block = resp.read(CHUNK)
                    if not block:
                        break
                    out.write(block)
                    got += len(block)
            if declared is not None and got != declared:
                raise IOError(f"short read: {got} of {declared} bytes")
            secs = time.time() - t0
            tmp.replace(dest)
            digest = sha256_of(dest)
            log(f"  ok     {key} ({got:,} bytes, {secs:.0f}s)")
            return {"url": url, "bytes": got, "sha256": digest,
                    "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        except (urllib.error.URLError, IOError, TimeoutError) as exc:
            last_err = exc
            tmp.unlink(missing_ok=True)
            if attempt < RETRIES:
                wait = 5 * attempt
                log(f"  retry  {key} after {type(exc).__name__}: {exc} (sleep {wait}s)")
                time.sleep(wait)

    raise SystemExit(f"FATAL: could not download {url}: {last_err}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--what", choices=["acs", "pums", "xwalk", "all"], default="all")
    ap.add_argument("--year", type=int, default=2023,
                    help="ACS 5-year vintage (default 2023, matching LODES8 2023)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--states", default=",".join(PUMS_STATES),
                    help="comma-separated PUMS state postal codes")
    ap.add_argument("--force", action="store_true", help="redownload even if verified")
    args = ap.parse_args()

    out = args.out
    manifest_path = out / "manifest_acs_pums.json"
    manifest = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text()).get("files", {})

    jobs = []
    if args.what in ("acs", "all"):
        (out / "acs").mkdir(parents=True, exist_ok=True)
        base = ACS_BASE.format(year=args.year)
        for table in ACS_TABLES:
            name = f"acsdt5y{args.year}-{table}.dat"
            jobs.append((f"{base}/{name}", out / "acs" / name))
    if args.what in ("pums", "all"):
        (out / "pums").mkdir(parents=True, exist_ok=True)
        base = PUMS_BASE.format(year=args.year)
        for st in [s.strip().lower() for s in args.states.split(",") if s.strip()]:
            for kind in ("p", "h"):
                name = f"csv_{kind}{st}.zip"
                jobs.append((f"{base}/{name}", out / "pums" / name))
    if args.what in ("xwalk", "all"):
        (out / "xwalk").mkdir(parents=True, exist_ok=True)
        jobs.append((XWALK_URL, out / "xwalk" / "2020_tract_to_2020_puma.txt"))

    print(f"{len(jobs)} file(s) -> {out}")
    total = 0
    for url, dest in jobs:
        entry = download(url, dest, force=args.force, manifest=manifest)
        manifest[dest.name] = entry
        total += entry["bytes"]
        manifest_path.write_text(json.dumps(
            {"year": args.year, "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             "files": manifest}, indent=2, sort_keys=True))

    print(f"done: {len(jobs)} file(s), {total / 1e9:.2f} GB, manifest at {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

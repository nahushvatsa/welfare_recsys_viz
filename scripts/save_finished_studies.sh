#!/usr/bin/env bash
# Poll the backend and dump each study to disk the moment it becomes available.
#
# Studies live only in the uvicorn process's memory (RunManager caches 4, LRU),
# and nothing in the API or the frontend writes them to disk — so a restart or a
# fifth study loses them. This saves GET /api/runs/{id} (aggregated metrics +
# per-condition/seed index) as soon as available flips true.
#
# Read-only with respect to the runs: a GET only refreshes the LRU timestamp.
#
#   ./scripts/save_finished_studies.sh            # until every known run is saved
#
# Emits one line per save, per error, and one final line when nothing is left.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$REPO/data/study_exports"
API=http://127.0.0.1:8000
PY="$REPO/.venv/bin/python"
mkdir -p "$OUT"

while :; do
    runs=$(curl -fsS --max-time 20 "$API/api/runs" 2>/dev/null) || {
        echo "WARN: backend unreachable at $(date -u +%H:%M:%S)"; sleep 60; continue
    }

    # "<run_id> <city> <status> <available>" per line.
    rows=$(printf '%s' "$runs" | "$PY" -c "
import json,sys
for r in json.load(sys.stdin)['runs']:
    print(r['run_id'], r['city'], r['status'], r['available'])
" 2>/dev/null) || { echo 'WARN: unparseable /api/runs payload'; sleep 60; continue; }

    pending=0
    while read -r id city status avail; do
        [ -z "${id:-}" ] && continue
        dest="$OUT/${city}_${id}.json"
        if [ "$status" = "error" ]; then
            echo "ERROR: $city ($id) failed — $(printf '%s' "$runs" | "$PY" -c "
import json,sys
print(next((r.get('error') for r in json.load(sys.stdin)['runs'] if r['run_id']=='$id'), ''))" 2>/dev/null)"
            continue
        fi
        if [ "$avail" != "True" ]; then
            pending=$((pending + 1)); continue
        fi
        [ -s "$dest" ] && continue
        if curl -fsS --max-time 300 "$API/api/runs/$id" -o "$dest.part" 2>/dev/null \
           && [ -s "$dest.part" ]; then
            mv "$dest.part" "$dest"
            echo "SAVED $city -> $dest ($(du -h "$dest" | cut -f1))"
        else
            rm -f "$dest.part"
            echo "WARN: save failed for $city ($id); retrying next pass"
            pending=$((pending + 1))
        fi
    done <<< "$rows"

    if [ "$pending" -eq 0 ]; then
        echo "DONE: every known run is saved under $OUT"
        exit 0
    fi
    sleep 60
done

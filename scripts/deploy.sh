#!/usr/bin/env bash
# Manual deploy. Never triggered automatically — you run it, you watch it.
#
#   ./scripts/deploy.sh              # deploy the current branch
#   ./scripts/deploy.sh --no-pull    # rebuild+restart without touching git
#
# Steps: git pull -> venv deps -> frontend build -> publish to /var/www/html
#        -> restart welfare.service -> health check.
#
# Requires: nvm (Node 20; the system node is v12 and cannot build Vite 7) and
# passwordless-or-interactive sudo for the single systemctl restart.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WEB_ROOT=/var/www/html
SERVICE=welfare.service
cd "$REPO"

step() { printf '\n\033[1m== %s\033[0m\n' "$1"; }

if [[ "${1:-}" != "--no-pull" ]]; then
    step "git pull ($(git rev-parse --abbrev-ref HEAD))"
    git pull --ff-only
fi

step "python deps"
.venv/bin/pip install -e '.[server,postgres]' --quiet

step "frontend build (Node 20 via nvm)"
export NVM_DIR="${NVM_DIR:-$HOME/.nvm}"
# shellcheck disable=SC1091
. "$NVM_DIR/nvm.sh"
nvm use 20 >/dev/null
cd frontend
npm ci --silent
npm run build            # tsc --noEmit && vite build; fails loudly on type errors
cd "$REPO"

step "publish to $WEB_ROOT"
# --delete keeps the web root an exact mirror of dist/, so removed bundles do
# not linger. _old/ holds the pre-existing nginx pages and is excluded.
rsync -a --delete --exclude '_old' frontend/dist/ "$WEB_ROOT/"

step "restart $SERVICE"
sudo systemctl restart "$SERVICE"

step "health check"
for i in $(seq 1 15); do
    if curl -fsS --max-time 5 http://127.0.0.1:8000/api/health >/dev/null 2>&1; then
        echo "  backend healthy after ${i}s"
        curl -fsS http://127.0.0.1:8000/api/health; echo
        echo
        echo "Deployed. https://airecsim.cusp.nyu.edu/"
        exit 0
    fi
    sleep 1
done

echo "  BACKEND DID NOT COME UP — recent logs:" >&2
sudo journalctl -u "$SERVICE" -n 30 --no-pager >&2
exit 1

#!/usr/bin/env bash
set -euo pipefail

ROOT="${RETENTION_PROJECT_ROOT:-$(pwd)}"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export RETENTION_API_PORT="${RETENTION_API_PORT:-8010}"
export RETENTION_AUTOOPS_API_BASE_URL="${RETENTION_AUTOOPS_API_BASE_URL:-http://localhost:${RETENTION_API_PORT}}"

if [ ! -d ".venv" ]; then
  python -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python scripts/run_universal_autoops_api.py &
API_PID=$!
trap 'kill $API_PID 2>/dev/null || true' EXIT
sleep 2
streamlit run dashboard/app.py

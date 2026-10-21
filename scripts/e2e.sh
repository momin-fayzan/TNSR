#!/usr/bin/env bash
# Full-stack end-to-end test: train a tiny model from scratch in a throwaway TNSR_HOME,
# start the real backend serving the built UI, and drive the browser with Playwright.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PORT="${TNSR_E2E_PORT:-8765}"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/tnsr-e2e.XXXXXX")"
export TNSR_HOME="$WORK" TNSR_DB="$WORK/var/e2e.db" TNSR_DEVICE=cpu
SERVER_PID=""
cleanup() {
  [[ -n "$SERVER_PID" ]] && kill "$SERVER_PID" 2>/dev/null || true
  rm -rf "$WORK"
}
trap cleanup EXIT

echo "==> Preparing data and training a tiny model in $WORK"
uv run tnsr data prepare --name demo --docs 3000 --vocab-size 800 >/dev/null
uv run tnsr train --config tiny --set train.max_steps=300 --set train.eval_interval=100 \
  --set model.block_size=64 --set model.n_embd=64 --set train.learning_rate=1e-2 | tail -3

echo "==> Building the UI"
npm --workspace tnsr_ui run build >/dev/null

echo "==> Starting backend on :$PORT"
uv run tnsr serve --port "$PORT" >"$WORK/server.log" 2>&1 &
SERVER_PID=$!
for _ in $(seq 1 100); do
  curl -sf "http://127.0.0.1:$PORT/api/health" >/dev/null && break
  sleep 0.2
done
curl -sf "http://127.0.0.1:$PORT/api/health" >/dev/null || { cat "$WORK/server.log"; exit 1; }

echo "==> Running Playwright"
TNSR_E2E_URL="http://127.0.0.1:$PORT" npm --workspace tnsr_ui run e2e

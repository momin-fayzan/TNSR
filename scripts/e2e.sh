#!/usr/bin/env bash
# Full-stack end-to-end test: label positions with Stockfish and train a tiny chess model from
# scratch in a throwaway TNSR_HOME,
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

echo "==> Labeling a few positions with Stockfish and training a tiny chess model in $WORK"
cat > "$WORK/games.pgn" <<'PGN'
[Event "e2e"]
[Site "https://lichess.org/e2e00001"]
[Result "1-0"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 d6 8. c3 O-O 1-0

[Event "e2e"]
[Site "https://lichess.org/e2e00002"]
[Result "0-1"]

1. d4 Nf6 2. c4 e6 3. Nc3 Bb4 4. e3 O-O 5. Bd3 d5 6. Nf3 c5 7. O-O Nc6 8. a3 Bxc3 0-1

[Event "e2e"]
[Site "https://lichess.org/e2e00003"]
[Result "1/2-1/2"]

1. c4 e5 2. Nc3 Nf6 3. g3 d5 4. cxd5 Nxd5 5. Bg2 Nb6 6. Nf3 Nc6 7. O-O Be7 8. d3 O-O 1/2-1/2
PGN
uv run tnsr chess prepare "$WORK/games.pgn" --name chess-demo --max-positions 40 --per-game 14 --nodes 150 --val-fraction 0.34 >/dev/null
uv run tnsr train --config chess-tiny --set train.max_steps=60 --set train.eval_interval=30 | tail -2

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

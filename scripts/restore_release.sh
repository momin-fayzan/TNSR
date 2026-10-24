#!/usr/bin/env bash
# Rebuild experiments/<run>/ from the run records and best checkpoints committed under
# releases/runs/, so Play and the Lab work without retraining. Existing run folders are never
# touched. Released checkpoints are inference-only (no optimizer state).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HOME_DIR="${TNSR_HOME:-$ROOT}"
cd "$ROOT"

for src in "$ROOT"/releases/runs/*/; do
  [[ -f "$src/run.json" ]] || continue
  run="$(basename "$src")"
  dest="$HOME_DIR/experiments/$run"
  if [[ -e "$dest" ]]; then echo "skip $run (already exists)"; continue; fi
  best="$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['best'])" "$src/checkpoints.json")"
  tmp="$(mktemp -d)"
  tar -xzf "$src/best-checkpoint.tar.gz" -C "$tmp"
  mkdir -p "$dest/checkpoints"
  cp "$src"/{run.json,status.json,metrics.jsonl} "$dest/"
  for kind in evals benchmarks; do
    [[ -d "$src/$kind" ]] && cp -r "$src/$kind" "$dest/"
  done
  mv "$tmp/checkpoint" "$dest/checkpoints/$best"
  rm -rf "$tmp"
  # Only the best checkpoint is shipped, so it is both best and latest.
  python3 - "$src/checkpoints.json" "$dest/checkpoints.json" "$best" <<'PY'
import json, sys
index = json.load(open(sys.argv[1]))
best = sys.argv[3]
index["checkpoints"] = [e for e in index["checkpoints"] if e["name"] == best]
index["latest"] = best
json.dump(index, open(sys.argv[2], "w"), indent=2)
PY
  echo "restored $run ($best)"
done

# Small released datasets (e.g. positions from TNSR's own games, labeled by Stockfish).
for archive in "$ROOT"/releases/datasets/*.tar.gz; do
  [[ -f "$archive" ]] || continue
  name="$(basename "$archive" .tar.gz)"
  dest="$HOME_DIR/datasets/$name"
  if [[ -e "$dest" ]]; then echo "skip dataset $name (already exists)"; continue; fi
  mkdir -p "$HOME_DIR/datasets"
  tar -xzf "$archive" -C "$HOME_DIR/datasets"
  echo "restored dataset $name"
done

if [[ ! -f "$HOME_DIR/datasets/chess-puzzles/puzzles.csv" ]]; then
  echo "To evaluate models, download puzzles once: uv run tnsr chess puzzles --count 5000"
fi
echo "Lichess and ChessBench training data are not shipped; make them with: uv run tnsr chess prepare --month 2013-01 --name chess-v1 --max-positions 60000 --endgames 15000 && uv run tnsr chess import-chessbench --files 0 --name chessbench-f0"

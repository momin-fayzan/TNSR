#!/usr/bin/env bash
# Rebuild experiments/<run>/ from the records and best checkpoints committed under releases/runs/,
# so Studio and Lab work without retraining. Existing run folders are never touched.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HOME_DIR="${TNSR_HOME:-$ROOT}"
# The datasets are deterministic, so regenerating them reproduces the exact token files the
# released runs were trained on (needed for evaluation and further training, not for chat).
cd "$ROOT"
[[ -f "$HOME_DIR/datasets/demo/meta.json" ]] || uv run tnsr data prepare --name demo >/dev/null
[[ -f "$HOME_DIR/datasets/demo-char/meta.json" ]] || uv run tnsr data prepare --name demo-char --tokenizer char >/dev/null
for name in demo demo-char; do
  echo "dataset $name: $(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['identity'])" "$HOME_DIR/datasets/$name/meta.json")"
done

for src in "$ROOT"/releases/runs/*/; do
  run="$(basename "$src")"
  dest="$HOME_DIR/experiments/$run"
  if [[ -e "$dest" ]]; then echo "skip $run (already exists)"; continue; fi
  best="$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['best'])" "$src/checkpoints.json")"
  tmp="$(mktemp -d)"
  tar -xzf "$src/best-checkpoint.tar.gz" -C "$tmp"
  mkdir -p "$dest/checkpoints"
  cp "$src"/{run.json,status.json,metrics.jsonl} "$dest/"
  cp -r "$src/evals" "$src/benchmarks" "$dest/" 2>/dev/null || true
  mv "$tmp/checkpoint" "$dest/checkpoints/$best"
  rm -rf "$tmp"
  # Only the best checkpoint is shipped (without optimizer state), so it is both best and latest.
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

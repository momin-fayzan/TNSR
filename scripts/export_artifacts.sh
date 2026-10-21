#!/usr/bin/env bash
# Bundle valuable artifacts (checkpoints, experiment records, dataset metadata, conversations DB)
# into exports/tnsr-artifacts-<timestamp>.tar.gz. Never deletes or modifies the originals.
#   scripts/export_artifacts.sh                 best checkpoint of every run (inference files only)
#   scripts/export_artifacts.sh --train-state   also include optimizer state so runs can be resumed elsewhere
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
HOME_DIR="${TNSR_HOME:-$ROOT}"
STAMP="$(date +%Y%m%d-%H%M%S)"
OUT="$ROOT/exports/tnsr-artifacts-$STAMP"
mkdir -p "$OUT/checkpoints" "$OUT/experiments" "$OUT/datasets"
FLAG=""
[[ "${1:-}" == "--train-state" ]] && FLAG="--include-train-state"

for run in "$HOME_DIR"/experiments/*/; do
  [[ -f "$run/run.json" ]] || continue
  id="$(basename "$run")"
  mkdir -p "$OUT/experiments/$id"
  # Small, human-readable records: config, environment, metrics, evaluations, benchmarks.
  cp "$run"/run.json "$run"/status.json "$run"/metrics.jsonl "$run"/checkpoints.json "$OUT/experiments/$id/" 2>/dev/null || true
  for kind in evals benchmarks; do
    [[ -d "$run/$kind" ]] && cp -r "$run/$kind" "$OUT/experiments/$id/"
  done
  if uv run tnsr models export "$id" "$OUT/checkpoints/$id.tar.gz" $FLAG >/dev/null 2>&1; then
    echo "exported checkpoint of $id"
  else
    echo "skipped $id (no checkpoint)"
  fi
done
for ds in "$HOME_DIR"/datasets/*/; do
  [[ -f "$ds/meta.json" ]] || continue
  name="$(basename "$ds")"
  mkdir -p "$OUT/datasets/$name"
  cp "$ds/meta.json" "$ds/tokenizer.json" "$OUT/datasets/$name/"
done
if [[ -f "$HOME_DIR/var/tnsr.db" ]]; then
  # Consistent copy of the SQLite database even while the server is running.
  python3 -c "import sqlite3,sys; s=sqlite3.connect(sys.argv[1]); d=sqlite3.connect(sys.argv[2]); s.backup(d)" \
    "$HOME_DIR/var/tnsr.db" "$OUT/tnsr.db"
fi
tar -czf "$OUT.tar.gz" -C "$ROOT/exports" "$(basename "$OUT")"
rm -rf "$OUT"
echo "Wrote $OUT.tar.gz ($(du -h "$OUT.tar.gz" | cut -f1))"
echo "Datasets are regenerated with 'tnsr data prepare' (synthetic) or from your own corpus; token files are not bundled."

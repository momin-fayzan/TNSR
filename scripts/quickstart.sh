#!/usr/bin/env bash
# From a fresh clone to a trained demo model and a running app.
#   scripts/quickstart.sh            full demo model (~30 min on a 4-core CPU, faster on Apple Silicon)
#   scripts/quickstart.sh --quick    small 3-minute model to try the workflow
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
QUICK=0
[[ "${1:-}" == "--quick" ]] && QUICK=1

command -v uv >/dev/null || { echo "uv is required: https://docs.astral.sh/uv/getting-started/installation/"; exit 1; }
command -v npm >/dev/null || { echo "Node.js 20+ with npm is required: https://nodejs.org"; exit 1; }

echo "==> Installing dependencies"
uv sync --extra dev
npm install

echo "==> Environment diagnostics"
uv run tnsr doctor --quick || true

if [[ ! -f datasets/demo/meta.json ]]; then
  echo "==> Preparing the synthetic demo dataset"
  uv run tnsr data prepare --name demo
fi

echo "==> Training"
if [[ $QUICK == 1 ]]; then
  uv run tnsr train --config demo --name quick --set train.max_steps=400 --set train.eval_interval=100 \
    --set model.n_layer=4 --set model.n_embd=128 --set model.n_head=4
else
  uv run tnsr train --config demo
fi

echo "==> Evaluating and benchmarking the best checkpoint"
uv run tnsr evaluate
uv run tnsr benchmark

echo "==> Building the UI"
npm run build

echo
echo "Done. Start the app with:  uv run tnsr serve   then open http://127.0.0.1:8000"

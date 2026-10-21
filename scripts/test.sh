#!/usr/bin/env bash
# Single entry point for all checks.
#   scripts/test.sh            lint + types + Python tests + UI tests (fast, deterministic)
#   scripts/test.sh --slow     also run the longer training/benchmark tests
#   scripts/test.sh --e2e      also run the full-stack browser test (trains a tiny model)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
SLOW=0
E2E=0
for arg in "$@"; do
  case "$arg" in
    --slow) SLOW=1 ;;
    --e2e) E2E=1 ;;
    *) echo "unknown option $arg" >&2; exit 2 ;;
  esac
done

step() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }

step "ruff (lint + format check)"
uv run ruff check .
uv run ruff format --check .
step "mypy"
uv run mypy
step "pytest"
if [[ $SLOW == 1 ]]; then uv run pytest -m "" ; else uv run pytest; fi
step "UI: eslint + typecheck"
npm --workspace tnsr_ui run lint
npm --workspace tnsr_ui run typecheck
step "UI: vitest"
npm --workspace tnsr_ui run test
if [[ $E2E == 1 ]]; then
  step "end-to-end (Playwright)"
  bash scripts/e2e.sh
fi
step "all checks passed"

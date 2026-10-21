# Working on TNSR

Guidance for contributors and coding agents.

## Layout

- `tnsr_core/` — ML code (no web dependencies). Packages: `tokenizer`, `model`, `data`,
  `training`, `inference`, `evaluation`, `checkpoints`, `config` (+ TOML presets), `cli.py`.
- `tnsr_server/` — FastAPI app (`app.py` factory), `api/` routes + schemas, `services/`
  (models, generation, jobs, experiments), `persistence/db.py` (SQLite + migrations).
- `tnsr_ui/` — React/TypeScript/Vite/Tailwind app; `src/lib/api.ts` mirrors the API.
- `tests/` — pytest (fast by default; `-m slow` for long runs). UI tests live next to the code
  (`*.test.ts(x)`); Playwright E2E in `tnsr_ui/e2e/`, driven by `scripts/e2e.sh`.
- Generated artifacts (`experiments/`, `datasets/`, `checkpoints/`, `var/`, `exports/`) are
  git-ignored.

## Commands

```bash
uv sync --extra dev && npm install     # setup
scripts/test.sh                        # everything that runs in CI (lint, types, pytest, vitest)
scripts/test.sh --slow --e2e           # plus long training tests and the browser E2E
npm run dev                            # API on :8000 (reload) + Vite on :5173
uv run tnsr --help
```

## Conventions

- Python ≥ 3.10, ruff (line length 100) + mypy must pass. Keep `tnsr_core` importable without
  FastAPI.
- Never fabricate metrics or results: every number shown in the UI must come from a file
  written by the pipeline.
- Never overwrite user artifacts: checkpoint and dataset writes are atomic and refuse to
  replace existing paths; run directories are always new.
- Any user-supplied path or id goes through `tnsr_core.paths.validate_name` / `safe_join` or
  `checkpoints.resolve_checkpoint_id` (confined to `TNSR_HOME`).
- Changing the checkpoint, tokenizer or dataset layout requires bumping its `*_FORMAT_VERSION`
  and keeping loading errors explicit. Schema changes require a new entry in
  `persistence/db.py::MIGRATIONS` (never edit an applied migration).
- Keep the API schemas (`tnsr_server/api/schemas.py`) and `tnsr_ui/src/lib/types.ts` in sync.
- Progress and state for long-running work is tracked in `docs/progress.md`.

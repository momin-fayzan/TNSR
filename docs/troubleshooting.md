# Troubleshooting

Start with `uv run tnsr doctor` — it checks Python, PyTorch, the selected device (with a
forward/backward smoke test), memory, disk, write access to `TNSR_HOME`, Node.js, the UI build
and prepared datasets. The Settings page shows the same diagnostics.

## Installation

| symptom | fix |
|---|---|
| `uv: command not found` | install uv: `curl -LsSf https://astral.sh/uv/install.sh \| sh` (or `brew install uv`) |
| `uv sync` downloads a huge CUDA wheel on Linux | the lockfile pins the CPU wheel index for Linux; run `uv sync --extra dev` from the repo root so `pyproject.toml`'s `[tool.uv.sources]` applies |
| want CUDA on Linux | `uv pip install torch --index-url https://download.pytorch.org/whl/cu124` inside `.venv` (outside the lockfile) |
| `npm install` fails on old Node | use Node.js 20 or newer |

## Apple Silicon / macOS

- `tnsr doctor` should report `device: mps`. If it reports CPU on an M-series Mac, check that
  you are running a native arm64 Python (`python -c "import platform; print(platform.machine())"`
  must print `arm64`, not `x86_64` under Rosetta).
- `NotImplementedError: ... not currently implemented for the MPS device`: run with
  `PYTORCH_ENABLE_MPS_FALLBACK=1`, or force CPU with `--device cpu` (training) /
  `TNSR_DEVICE=cpu` (server).
- MPS throughput for very small models can be lower than CPU because of kernel launch
  overhead; benchmark both with `tnsr benchmark --device cpu` and `--device mps`.

## Training

| symptom | fix |
|---|---|
| `Dataset 'demo' not found` | `uv run tnsr data prepare` |
| `split has only N tokens; need more than block_size+1` | use a smaller `model.block_size` or a bigger corpus |
| `Aborting: N consecutive non-finite steps` | lower `train.learning_rate`, keep `grad_clip = 1.0`, use `precision = "fp32"` |
| out of memory | lower `train.batch_size` and raise `train.grad_accum_steps` to keep the effective batch |
| run shows `running` but nothing happens | the process died without updating `status.json` (e.g. `kill -9`); resume it: `tnsr train --resume <run>` |
| `Dataset changed since this run started` | the dataset was re-prepared with different content; start a new run |

## Server and UI

| symptom | fix |
|---|---|
| UI shows "Backend offline" | start the API: `uv run tnsr serve` (port 8000). In dev mode Vite proxies `/api` to `TNSR_API` |
| `http://127.0.0.1:8000` shows JSON 404 for `/` | the UI isn't built: `npm run build` (or use `npm run dev` and open :5173) |
| "No trained models found" | train one (`uv run tnsr train --config demo`) or import an archive (`uv run tnsr models import file.tar.gz --name demo`) |
| "Model unavailable: … not found" | the conversation points at a deleted checkpoint; pick another model in the header |
| "The model is busy with another generation" | one generation runs at a time; wait or stop the other one |
| port 8000 in use | `uv run tnsr serve --port 8010` and set `TNSR_API=http://127.0.0.1:8010` for `npm run dev` |
| need access from another device | `--host 0.0.0.0` works but the API has **no authentication**; only do this on a trusted network |

## Recovering data

- Conversations: `var/tnsr.db` (SQLite). `scripts/export_artifacts.sh` takes a consistent backup
  even while the server runs.
- Checkpoints: `uv run tnsr models export <run-or-checkpoint> out.tar.gz [--include-train-state]`
  then `uv run tnsr models import out.tar.gz --name <name>` on another machine.
- A newer TNSR database refuses to open with an older TNSR build instead of being downgraded;
  upgrade TNSR.
- Interrupted checkpoint writes leave `*.tmp` directories, which are ignored and can be deleted.

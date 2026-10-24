# Progress log

Read this to pick up the work after an interruption. Newest state first.

## Current state (2026-10-24): chess engine

TNSR is now a transformer chess engine without search (see `docs/chess.md`, ADR 0003). The
text-LM milestones below remain done and tested; the product is chess.

Done and verified on the dev VM:
- Chess core: tokenizer with move marks (v2 format; v1 checkpoints still load), value head
  with mate-distance classes, Stockfish labeling (resumable, parallel), generated endgames,
  on-policy game collection (`tnsr chess collect`), ChessBench import
  (`tnsr chess import-chessbench`, file 0 imported: 16.7M records), combined datasets
  (`data.dataset = "a+b"`), shared-board mode (`data.shared_board`, kept as an option; it loses
  to marks, see the experiment log), optional rules-aware scoring (`--rules`).
- Engine, UCI, evaluation (held-out move quality, puzzles, Stockfish matches, random mover).
- Play UI and chess API; Lab shows chess runs and evaluations.
- Released models: `releases/runs/chess-v1-mini-*`, `chess-v2-mini-*` (restore with
  `scripts/restore_release.sh`). Best: v2, 26.5% top-1, 9.6% puzzles (25.2% with `--rules`),
  ~950 Elo on Stockfish's scale (50 games per level), beats a random mover.

In progress / next:
- Owner is training `chess-small` (6.9M) on `chess-v1+chessbench-f0` on an M-series MacBook Air.
- Decisions pending with the owner: `--rules` on by default; GPU budget; Lichess bot account
  for a real rating.
- Next experiments: more ChessBench files with longer training (compute-bound on CPU); a second
  on-policy round from v2; larger models once on Apple Silicon/GPU.

## Current state (2026-10-21)

All milestones (0–8) are implemented and verified in the development environment: Linux VM,
4 x86 cores, 16 GB RAM, no GPU, Python 3.12, PyTorch 2.14.1+cpu, Node 22.

| milestone | status | how it was verified |
|---|---|---|
| 0 Environment & architecture | done | `tnsr doctor`; uv + npm lockfiles |
| 1 Model, tokenizers, loss | done | `tests/test_model.py`, `tests/test_tokenizer.py` |
| 2 Data, training, checkpoints, resume | done | `tests/test_training.py` (bit-exact resume); demo run of 4000 steps; char run interrupted with SIGINT at step 257 and resumed via CLI |
| 3 Inference, evaluation, benchmarks | done | `tests/test_inference.py`, `tests/test_evaluation_cli.py`; real eval and benchmark results in `releases/runs/*` |
| 4 Backend | done | `tests/test_server.py`, `tests/test_integration.py` (real uvicorn) |
| 5 Studio frontend | done | Vitest suites; manual screenshots; Playwright E2E |
| 6 Lab | done | E2E and screenshots against the real runs (compare view flags tokenizer mismatch) |
| 7 Integration & reliability | done | `scripts/e2e.sh`: tiny model trained from scratch → server → browser streaming, stop, reload, rename, Lab |
| 8 Release quality | done | README, docs/, scripts, `scripts/test.sh --slow --e2e` green |

## Definition-of-done checklist

1. Install deps: `uv sync --extra dev`, `npm install` ✔
2. Diagnostics: `uv run tnsr doctor` ✔
3. Demo dataset: `uv run tnsr data prepare` ✔ (identity `demo-b1ee33d9ce21`, 1.24M train / 65.7K val tokens)
4. Train from scratch: `uv run tnsr train --config demo` ✔ (3.02M params, 4000 steps)
5. Real metrics: train 0.380 / val 0.398 at step 4000; best val 0.396 at step 3500 ✔
6. Save and resume: char run interrupted at step 257 and resumed to 1500 ✔; bit-exactness is
   covered by a test
7. Independent inference engine loads the weights ✔ (`tnsr generate`, `InferenceEngine`)
8. Generation without an external API ✔
9. Backend and frontend start: `uv run tnsr serve` serving the built UI ✔
10. Send a message in the app ✔ (Playwright and manual screenshots)
11. Streaming visible in the UI ✔ (MutationObserver assertion in the E2E)
12. Cancel a generation ✔ (UI stop button, cancel endpoint, client disconnect)
13. Conversations persist across a reload ✔
14. Lab shows real training and evaluation results ✔
15. Automated test suite ✔ (`scripts/test.sh --slow --e2e`)
16. README reproduces the workflow ✔ (commands above were all run in this environment)
17. Artifacts preserved ✔: code is pushed to git; best checkpoints and run records are in
    `releases/runs/` (restore with `scripts/restore_release.sh`);
    `scripts/export_artifacts.sh` bundles everything

## Results snapshot

| | demo (BPE 1621) | char (73) |
|---|---|---|
| params | 3.02M | 0.83M |
| full-val loss / ppl | 0.3994 / 1.491 | 0.1852 / 1.203 |
| bits/byte | 0.1486 | 0.2682 |
| probe accuracy | 85.1% | 26.5% |
| decode tok/s, KV cache on/off | 673 / 213 | 1047 / 589 |
| TTFT (16-token prompt) | 2.3 ms | 1.3 ms |

## Verified on Apple Silicon (owner's MacBook Air)

Setup from the README, `tnsr doctor` (MPS detected), the app on port 8010 with chat streaming,
MPS and CPU smoke training (identical losses), and MPS vs CPU benchmarks. Findings and the
resulting defaults: MPS trains ~1.6× faster than CPU, but the CPU generates ~3.5× faster for
these small models, so inference now uses the CPU for models under 50M parameters. The app
also now defaults to the best-evaluated model, and `tnsr serve` explains port conflicts.

## Not verified (environment limits)

- **CUDA**: not available here; code paths (autocast, fused AdamW, CUDA RNG state) are untested.
- **Windows**: not tested (signal handling uses SIGINT/SIGTERM; scripts are bash).

## Next actions (see roadmap)

Instruction fine-tuning stage; larger public-domain corpus recipe; `torch.compile` benchmarking on
accelerators; concurrent generation.

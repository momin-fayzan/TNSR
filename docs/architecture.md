# Architecture

TNSR is a monorepo with three layers that share one checkpoint format.

```
                ┌──────────────── TNSR Studio / Lab (tnsr_ui) ────────────────┐
                │ React + TypeScript + Tailwind, built by Vite                 │
                │ fetch() + Server-Sent Events over POST for streaming         │
                └───────────────────────────────┬──────────────────────────────┘
                                                │ /api (JSON, SSE)
                ┌───────────────────────────────▼──────────────────────────────┐
                │ tnsr_server (FastAPI, localhost only)                         │
                │  api/routes.py       HTTP + SSE endpoints, validation         │
                │  services/models     one loaded InferenceEngine + locks       │
                │  services/generation worker thread → asyncio queue, cancel    │
                │  services/jobs       subprocess jobs (train/eval/bench/prep)  │
                │  services/experiments  run folders → SQLite index             │
                │  persistence/db.py   SQLite, versioned migrations             │
                └───────────────┬───────────────────────────────┬──────────────┘
                     in-process │                     subprocess │ python -m tnsr_core.cli
                ┌───────────────▼───────────────────────────────▼──────────────┐
                │ tnsr_core                                                     │
                │  tokenizer/  char + byte-level BPE (trained here)             │
                │  model/      decoder-only Transformer, KV cache               │
                │  data/       corpus → doc-level split → token .bin files      │
                │  training/   Trainer: AdamW, schedule, eval, ckpt, resume     │
                │  inference/  sampling, streaming, stop strings                │
                │  evaluation/ loss / perplexity / bits-per-byte, benchmarks    │
                │  checkpoints/ format, atomic save, validated load, discovery  │
                │  cli.py      `tnsr` command                                   │
                └───────────────────────────────────────────────────────────────┘
```

## Data flow

1. **Data** — `tnsr data prepare` reads the synthetic generator or user text, splits *documents*
   into train/val by hashing `seed:document` (duplicates can't straddle the split), fits the
   tokenizer on the train split only, and writes `datasets/<name>/{train,val}.bin`
   (uint16/uint32, EOS after every document), `tokenizer.json` and `meta.json`
   (provenance, license, statistics, content-hash identity).
2. **Training** — `Trainer` samples random windows from the memory-mapped train stream,
   evaluates on fixed, evenly spaced windows (so losses are comparable across steps) and writes
   everything under `experiments/<run_id>/`:
   `run.json` (config, environment, dataset/tokenizer identity), `status.json` (live state,
   rewritten atomically), `metrics.jsonl` (append-only), `checkpoints/step_XXXXXXX/`,
   `checkpoints.json` (index with best/latest), `evals/`, `benchmarks/`.
3. **Inference** — `InferenceEngine` loads `model.pt` + `meta.json` + `tokenizer.json` only.
   It never touches the dataset or optimizer state.
4. **Serving** — the API loads one engine at a time; chat requests become prompts in the
   `User:` / `TNSR:` format used by the dialogue part of the corpus.

## Model

`tnsr_core/model/transformer.py`, pre-norm GPT-style decoder:

- token embedding + learned absolute position embedding (or RoPE, `pos_encoding = "rope"`)
- `n_layer` × [RMSNorm/LayerNorm → causal multi-head self-attention → residual,
  norm → GELU MLP (4×) → residual]
- final norm, vocabulary projection weight-tied to the token embedding
- attention via `F.scaled_dot_product_attention` (`attn_impl = "sdpa"`) or an explicit
  masked softmax (`"manual"`); both are tested to agree and to be causal
- GPT-2 initialisation with scaled residual projections; weight decay on matrices only
- per-layer KV cache for incremental decoding; generation past the context window rebuilds the
  cache from the most recent half-window

## Checkpoint format (version 1)

```
step_0004000/
  model.pt          state_dict only — loaded with torch.load(weights_only=True)
  meta.json         format_version, model_config, num_parameters, tokenizer identity,
                    step, tokens_seen, dataset identity, training settings, metrics
  tokenizer.json    exact tokenizer (format_version, kind, special tokens, merges/chars)
  train_state.pt    optimizer, step, elapsed time, RNG states (Python/NumPy/torch/CUDA,
                    data sampler) — only needed to resume
```

Saves are atomic (write `*.tmp`, then rename) and never overwrite. Loading validates the
format version, required files, tokenizer identity vs metadata, vocabulary size vs model
config, and that the state dict matches the architecture exactly.

Checkpoint ids are paths relative to `TNSR_HOME`, e.g.
`experiments/demo-20261021-062619/checkpoints/step_0004000`. A run id resolves to the run's
best checkpoint. API inputs are resolved strictly inside `TNSR_HOME` (no `..`, no absolute paths).

## Server concurrency

- **Generation**: each request runs `engine.stream()` in a worker thread that pushes events onto
  an `asyncio.Queue`. A per-engine lock serializes generations (one CPU/GPU, no benefit from
  interleaving); a waiting request times out with a clear error after 120 s.
- **Cancellation**: a `threading.Event` per generation is set by
  `POST /api/generations/{id}/cancel` or when the client disconnects (the SSE generator's
  `finally` block). The partial response is persisted with status `cancelled`.
- **Model swaps** take the generation lock, so a stream never sees a half-replaced engine.
- **Training and evaluation** run as subprocesses (`python -m tnsr_core.cli …`) with logs in
  `var/jobs/`. They survive server restarts; stopping sends SIGINT, and the trainer saves a
  checkpoint before exiting. Runs started from the CLI can be stopped from the Lab too (the
  pid in `status.json` is verified to be a TNSR training process before signalling).

## Persistence

SQLite (`var/tnsr.db`, WAL mode, foreign keys on) with migrations keyed on
`PRAGMA user_version`. Tables: `conversations`, `messages`, `settings`, `jobs`,
`experiments`, `evaluations`. Weights and metrics stay on disk; the `experiments` and
`evaluations` tables are an index re-synced from the filesystem, and entries whose files
disappeared are flagged `missing` instead of being deleted. Messages left `streaming` by a
crash are marked as errors at startup. A database from a newer TNSR version is refused rather
than downgraded.

## Streaming protocol

`POST /api/conversations/{id}/messages` (and `/retry`, `/api/generate`) respond with
`text/event-stream`:

```
event: start  data: {"generation_id", "user_message", "assistant_message"}
event: token  data: {"text": "..."}              # repeated
event: done   data: {"message": {...}, "stats": {...}}
event: error  data: {"detail": "...", "message": {...} | null}
```

Errors detectable before streaming (no model, invalid settings, unknown conversation) are
ordinary JSON HTTP errors. Text is decoded incrementally so multi-byte UTF-8 characters split
across byte-level BPE tokens are never emitted half-formed, and text that might be the start of
a stop string is held back until it can be resolved.

## Frontend

`tnsr_ui/src`: `lib/api.ts` (typed client + SSE parser), `hooks/useChat.ts` (optimistic
messages, streaming, stop, retry), `pages/` (Studio, Lab, run detail, compare, Models,
Settings), `components/` (UI primitives, charts). Lab pages are lazy-loaded so the charting
library doesn't slow down the chat. In production the FastAPI app serves `tnsr_ui/dist`; in
development Vite proxies `/api` to the backend.

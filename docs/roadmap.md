# Roadmap

## Done (v0.1)

- [x] Decoder-only Transformer from scratch: learned or rotary positions, RMSNorm/LayerNorm,
      fused or explicit causal attention, weight tying, KV cache
- [x] Character and byte-level BPE tokenizers trained in-repo, versioned and hashed
- [x] Synthetic, MIT-licensed TinyTales corpus; user corpora from .txt/.md files
- [x] Leak-free document-level splits; tokenizer fitted on train only
- [x] Training: AdamW, warmup + cosine/linear decay, clipping, grad accumulation, autocast,
      periodic eval, atomic checkpoints with rotation, bit-exact resume, graceful interrupt,
      non-finite detection, run metadata and environment capture
- [x] Inference: temperature, top-k, top-p, repetition penalty, greedy and seeded sampling,
      stop strings, EOS handling, streaming with UTF-8-safe detokenization, TTFT/throughput stats
- [x] Evaluation (loss, perplexity, bits-per-byte, samples) and benchmarks (TTFT, tokens/s,
      KV cache on/off, peak memory)
- [x] CLI: doctor, data prepare/inspect/list, train/resume, generate, evaluate, benchmark,
      models list/show/export/import, runs list, serve
- [x] FastAPI backend: SSE streaming, cancellation and disconnect handling, SQLite with
      migrations, subprocess jobs for training/eval/benchmark/dataset prep
- [x] Studio: conversations, streaming chat and completion modes, stop/retry/copy, model
      picker, validated generation settings, offline/empty/error states
- [x] Lab: live runs, loss/LR/throughput charts, checkpoints, evaluations, benchmarks,
      comparisons with comparability warnings, JSON/CSV export, new runs, dataset preparation
- [x] Tests: Python unit/integration (real server), slow training tests, Vitest component and
      page tests, Playwright full-stack E2E

## Chess (current focus)

- [x] Action-value transformer without search, move marks, mate classes, endgame curriculum
- [x] On-policy positions (`tnsr chess collect`), ChessBench import, combined datasets
- [x] Evaluation: held-out move quality, Lichess puzzles, Stockfish matches, random mover
- [ ] Scale: `chess-small` (6.9M) on several ChessBench files on Apple Silicon
- [ ] Scale further on a rented GPU (25–100M+ params, billions of records) — needs owner budget
- [ ] Iterate on-policy collection from each new model
- [ ] Real Lichess rating via a bot account (owner decision)
- [ ] Faster data path for ChessBench at scale (memory-mapped shards instead of loading all
      records into RAM)

## Planned (text LM, paused)

- [ ] **Instruction fine-tuning**: a supervised fine-tuning stage on prompt/response pairs with
      loss masked to the response tokens (`ignore_index` is already supported by the loss).
      Until then TNSR is a base model; its chat ability comes only from dialogue-formatted
      pretraining text.
- [ ] Larger public-domain corpus recipe (e.g. user-downloaded Project Gutenberg texts) with a
      documented provenance check, and a streaming tokenizer for corpora > 2 GB
- [ ] `torch.compile` profiling on CUDA/MPS (option exists; not yet benchmarked as a win)
- [ ] Batched / concurrent generation (currently one generation at a time per model)
- [ ] Hyperparameter sweeps from the Lab
- [ ] Safetensors weight format for cross-framework portability
- [ ] Multiple loaded models with LRU eviction
- [ ] Optional authentication for non-localhost deployments

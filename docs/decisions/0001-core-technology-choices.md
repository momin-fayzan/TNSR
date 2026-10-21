# ADR 0001: Core technology choices

**Status:** accepted · 2026-10-21

## Context

TNSR must train and run a language model from scratch on a laptop (including Apple Silicon)
without paid infrastructure, with a polished local UI.

## Decisions

1. **PyTorch tensors + autograd, model code written here.** No Hugging Face Transformers or other
   model zoo. `F.scaled_dot_product_attention` is used as a fused kernel, with an explicit
   masked-softmax path kept for verification (tests assert both agree and are causal).
2. **uv for Python, npm workspaces for the UI.** `uv.lock` pins every Python package; on Linux
   the CPU-only PyTorch index is used to avoid multi-GB CUDA downloads, while macOS gets the
   regular wheel (with MPS). `package-lock.json` pins the UI.
3. **Single local process + subprocess jobs**, not a task queue or microservices. Training is CPU/GPU
   bound and long-lived, so it runs as an independent `tnsr train` process that the API
   launches and observes through files. No Redis/Celery.
4. **Files are the source of truth for ML artifacts; SQLite for app state.** Checkpoints, metrics
   (JSONL) and results (JSON) are portable, inspectable and survive without the server. SQLite
   stores conversations, settings and job metadata, plus an index of experiments.
5. **SSE over POST for streaming.** Simpler than WebSockets for one-directional token streams;
   cancellation uses a separate endpoint plus client-disconnect detection.
6. **Synthetic training corpus.** Licensing of arbitrary web text is unclear, and large downloads
   are slow. A deterministic generator gives a reproducible, MIT-licensed corpus that a 3M model
   can learn visibly within ~30 minutes on a CPU, including a dialogue format for the chat UI.
7. **React + Vite + Tailwind + Recharts** for the UI; no component library, to keep the bundle
   small and the design consistent.

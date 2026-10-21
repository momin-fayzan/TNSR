# Evaluation and benchmarking

## Metrics

`uv run tnsr evaluate [MODEL]` scores the **entire validation split** (capped by
`--max-tokens`, default 200,000) with non-overlapping context windows, so every target token is
counted once:

| metric | definition | comparable across |
|---|---|---|
| `val_loss` | mean next-token cross-entropy (nats/token) | same tokenizer **and** same validation text |
| `perplexity` | `exp(val_loss)` | same tokenizer and validation text |
| `bits_per_byte` | total loss in bits / UTF-8 bytes of the scored text | any tokenizer, same validation text |

Perplexity depends on the tokenizer: a character model predicts many easy tokens per word, a
BPE model fewer, harder ones. TNSR therefore records the tokenizer identity, dataset identity and
a hash of the validation text with every result, and the Lab's compare view marks loss and
perplexity as *not comparable* whenever tokenizers differ, and every metric as not comparable
when the validation text differs. Bits-per-byte is the tokenizer-independent number to compare.

Bits-per-byte uses the validation split's average bytes per token, which is exact when the full
split is scored (`covered_full_split: true`).

Each evaluation also stores sample generations for fixed prompts (temperature 0.8, top-k 40,
fixed seeds) so qualitative changes are visible over time.

The training-time `val_loss` in `metrics.jsonl` uses `eval_iters × batch_size` fixed windows,
which is cheaper but a sample; `tnsr evaluate` is the authoritative number.

## Benchmarks

`uv run tnsr benchmark [MODEL]` measures, for prompts of 16 and 64 tokens, with and without the
KV cache (greedy decoding, EOS ignored, one warmup + `--repeats` timed runs, medians reported):

- **TTFT** — time from request to the first sampled token (tokenization + prefill + sampling)
- **decode tokens/s** — generated tokens after the first, per second
- **peak memory** — process RSS sampled every 5 ms (plus CUDA allocator peak / MPS driver
  memory where available)

Results are written to `experiments/<run>/benchmarks/` and shown in the Lab and on the Models
page. Benchmarks on different devices are flagged as not comparable.

## Profiling notes

Profiled with `cProfile` on the demo model (CPU, 4 threads). Decoding time goes to the model
forward pass: linear layers ~25%, attention ~10%, RMSNorm ~20%, and PyTorch module dispatch for
most of the rest. Sampling (top-k/top-p/multinomial) is about 5%.

- **KV cache, adopted:** it turns each decode step from O(context) into O(1) new positions, and
  the benchmark shows a **3.0–3.2× faster decode** on the demo model (673 vs 213 tokens/s).
- **Fused `F.rms_norm`, not adopted:** it gives identical outputs and saves about 2 µs per call,
  roughly 2% of per-token time, which is below run-to-run noise.
- **`torch.compile`:** available as `train.compile = true` but not enabled by default. Compile
  time dominates short CPU runs, and it hasn't been benchmarked as a win yet (see roadmap).

## Where results live

```
experiments/<run>/evals/<checkpoint>-<timestamp>.json
experiments/<run>/benchmarks/<checkpoint>-<timestamp>.json
checkpoints/<name>/{evals,benchmarks}/…     # for imported checkpoints
```

The server indexes these files into SQLite (`evaluations` table) and serves them at
`GET /api/evaluations`. Files are the source of truth; deleting one marks the index row missing.

## Interpreting the demo model

The TinyTales corpus is templated, so its entropy is low and a 3M-parameter model reaches a
validation loss well below what the same model would get on natural text. Low perplexity here
means "learned the generator's patterns", not general language ability. The val/train gap is
the useful overfitting signal: on the demo run the two stay within ~0.01 nats.

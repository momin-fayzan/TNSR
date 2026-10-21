"""Held-out evaluation (loss, perplexity, bits-per-byte) and inference benchmarks."""

from __future__ import annotations

import contextlib
import json
import math
import os
import statistics
import threading
import time
from pathlib import Path
from typing import Any

import psutil
import torch

from tnsr_core.checkpoints import checkpoint_id, read_meta, results_dir_for
from tnsr_core.data import TokenDataset
from tnsr_core.inference import GenerationConfig, InferenceEngine
from tnsr_core.training.runs import environment_info, write_json_atomic

DEFAULT_SAMPLE_PROMPTS = [
    "Once upon a time",
    "User: What is 3 plus 4?\nTNSR:",
    "User: What sound does a cow make?\nTNSR:",
    "The capital of France is",
]


def _save_result(ckpt_path: Path, kind: str, result: dict[str, Any]) -> Path:
    folder = results_dir_for(ckpt_path) / kind
    folder.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = folder / f"{ckpt_path.name}-{stamp}.json"
    i = 1
    while out.exists():
        out = folder / f"{ckpt_path.name}-{stamp}-{i}.json"
        i += 1
    write_json_atomic(out, result)
    return out


@torch.inference_mode()
def evaluate_checkpoint(
    ckpt_path: Path,
    dataset: str | None = None,
    device: str = "auto",
    max_tokens: int | None = 200_000,
    batch_size: int = 8,
    sample_prompts: list[str] | None = None,
    sample_tokens: int = 60,
    save: bool = True,
) -> dict[str, Any]:
    """Score the full (or capped) validation split with non-overlapping windows."""
    engine = InferenceEngine.from_checkpoint(ckpt_path, device)
    meta = read_meta(ckpt_path)
    ds_ref = dataset or (meta.get("dataset") or {}).get("name")
    if not ds_ref:
        raise ValueError("Checkpoint has no dataset reference; pass --dataset")
    ds = TokenDataset(ds_ref)
    if ds.tokenizer.identity != engine.tokenizer.identity:
        raise ValueError(
            f"Dataset {ds.name} uses tokenizer {ds.tokenizer.identity} but the checkpoint uses "
            f"{engine.tokenizer.identity}; token-level metrics would be meaningless. "
            "Prepare the dataset with the checkpoint's tokenizer instead."
        )
    model = engine.model
    block = model.config.block_size
    t0 = time.perf_counter()
    total_nats = 0.0
    total_tokens = 0
    for x, y in ds.sequential_batches("val", block, batch_size, max_tokens):
        x, y = x.to(engine.device), y.to(engine.device)
        logits, _ = model(x)
        loss = torch.nn.functional.cross_entropy(
            logits.float().view(-1, logits.size(-1)), y.reshape(-1), reduction="sum"
        )
        total_nats += loss.item()
        total_tokens += y.numel()
    if total_tokens == 0:
        raise ValueError("Validation split is smaller than one context window")
    mean_loss = total_nats / total_tokens
    stats = ds.meta["stats"]
    # Bytes covered ≈ tokens scored × average bytes/token of the val split.
    val_bytes_per_token = stats["val_bytes"] / max(1, stats["val_tokens"])
    bits_per_byte = (total_nats / math.log(2)) / (total_tokens * val_bytes_per_token)
    covered_all = total_tokens >= stats["val_tokens"] - block - 1

    samples = []
    for i, prompt in enumerate(
        sample_prompts if sample_prompts is not None else DEFAULT_SAMPLE_PROMPTS
    ):
        res = engine.generate(
            prompt,
            GenerationConfig(
                max_new_tokens=sample_tokens,
                deterministic=False,
                temperature=0.8,
                top_k=40,
                seed=i,
                stop=["\nUser:"],
            ),
        )
        samples.append(
            {"prompt": prompt, "completion": res.text, "finish_reason": res.stats.finish_reason}
        )

    result = {
        "kind": "evaluation",
        "checkpoint_id": checkpoint_id(ckpt_path),
        "checkpoint_name": ckpt_path.name,
        "step": meta.get("step"),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "dataset": {
            "name": ds.name,
            "identity": ds.identity,
            "val_text_sha256": ds.meta.get("val_text_sha256"),
            "split": "val",
        },
        "tokenizer": engine.tokenizer.describe(),
        "metrics": {
            "val_loss": round(mean_loss, 5),
            "perplexity": round(math.exp(min(mean_loss, 50)), 4),
            "bits_per_byte": round(bits_per_byte, 5),
            "tokens_evaluated": total_tokens,
            "covered_full_split": covered_all,
        },
        "comparability": {
            "perplexity": "only comparable between models with the same tokenizer identity on the same dataset",
            "bits_per_byte": "comparable across tokenizers when evaluated on the same validation text",
        },
        "seconds": round(time.perf_counter() - t0, 2),
        "device": str(engine.device),
        "samples": samples,
    }
    if save:
        result["path"] = str(_save_result(ckpt_path, "evals", result))
    return result


class _PeakRSS:
    """Samples process RSS in a background thread to capture the peak during a benchmark."""

    def __init__(self, interval: float = 0.005):
        self.proc = psutil.Process(os.getpid())
        self.interval = interval
        self.peak = self.proc.memory_info().rss
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            self.peak = max(self.peak, self.proc.memory_info().rss)
            time.sleep(self.interval)

    def __enter__(self) -> _PeakRSS:
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join()
        self.peak = max(self.peak, self.proc.memory_info().rss)


def benchmark_checkpoint(
    ckpt_path: Path,
    device: str = "auto",
    prompt_tokens: tuple[int, ...] = (16, 64),
    new_tokens: int = 64,
    repeats: int = 3,
    compare_no_cache: bool = True,
    save: bool = True,
) -> dict[str, Any]:
    """Measure time-to-first-token, decode throughput and peak memory."""
    engine = InferenceEngine.from_checkpoint(ckpt_path, device)
    tok = engine.tokenizer
    block = engine.block_size
    base_text = "One day, a little fox named Pip was in the forest under the tall pine trees. " * 40
    base_ids = tok.encode(base_text)
    if engine.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    cfg = GenerationConfig(max_new_tokens=new_tokens, deterministic=True, stop_at_eos=False)
    rows = []
    modes = [True, False] if compare_no_cache else [True]
    with _PeakRSS() as mem:
        for n_prompt in prompt_tokens:
            n_prompt = min(n_prompt, block - 1 - 1)
            prompt = tok.decode(base_ids[:n_prompt])
            n_new = min(new_tokens, block - n_prompt - 1)
            cfg.max_new_tokens = max(1, n_new)
            for use_cache in modes:
                engine.generate(prompt, cfg, use_cache=use_cache, add_bos=False)  # warmup
                ttfts, rates, totals = [], [], []
                for _ in range(repeats):
                    res = engine.generate(prompt, cfg, use_cache=use_cache, add_bos=False)
                    ttfts.append(res.stats.time_to_first_token_ms or 0.0)
                    rates.append(res.stats.decode_tokens_per_sec or 0.0)
                    totals.append(res.stats.total_ms)
                rows.append(
                    {
                        "prompt_tokens": res.stats.prompt_tokens,
                        "new_tokens": res.stats.generated_tokens,
                        "kv_cache": use_cache,
                        "ttft_ms_median": round(statistics.median(ttfts), 2),
                        "decode_tokens_per_sec_median": round(statistics.median(rates), 1),
                        "total_ms_median": round(statistics.median(totals), 1),
                    }
                )
    peak = {"process_peak_rss_mb": round(mem.peak / 1e6, 1)}
    if engine.device.type == "cuda":
        peak["cuda_peak_allocated_mb"] = round(torch.cuda.max_memory_allocated() / 1e6, 1)
    if engine.device.type == "mps" and hasattr(torch, "mps"):
        with contextlib.suppress(Exception):
            peak["mps_driver_allocated_mb"] = round(torch.mps.driver_allocated_memory() / 1e6, 1)
    meta = read_meta(ckpt_path)
    env = environment_info(str(engine.device), "benchmark")
    result = {
        "kind": "benchmark",
        "checkpoint_id": checkpoint_id(ckpt_path),
        "checkpoint_name": ckpt_path.name,
        "step": meta.get("step"),
        "num_parameters": meta.get("num_parameters"),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "device": str(engine.device),
        "threads": torch.get_num_threads(),
        "repeats": repeats,
        "results": rows,
        "memory": peak,
        "environment": {
            k: env[k]
            for k in ("platform", "machine", "python", "torch", "tnsr_version", "git_commit")
        },
    }
    if save:
        result["path"] = str(_save_result(ckpt_path, "benchmarks", result))
    return result


def load_results(root: Path, kind: str) -> list[dict[str, Any]]:
    folder = root / kind
    if not folder.is_dir():
        return []
    out = []
    for f in sorted(folder.glob("*.json")):
        try:
            out.append(json.loads(f.read_text()))
        except json.JSONDecodeError:
            continue
    return out

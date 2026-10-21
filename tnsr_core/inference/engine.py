"""Autoregressive inference with KV caching, stop strings and incremental streaming."""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from tnsr_core.checkpoints import LoadedCheckpoint, checkpoint_id, load_checkpoint
from tnsr_core.device import select_inference_device

from .sampling import GenerationConfig, select_token


@dataclass
class GenerationStats:
    prompt_tokens: int = 0
    generated_tokens: int = 0
    prompt_truncated: bool = False
    time_to_first_token_ms: float | None = None
    total_ms: float = 0.0
    decode_tokens_per_sec: float | None = None
    tokens_per_sec: float | None = None
    finish_reason: str = ""
    cache_rebuilds: int = 0

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class GenerationEvent:
    """``kind`` is 'token' (``text`` is new output) or 'done' (``stats`` is final)."""

    kind: str
    text: str = ""
    token_id: int | None = None
    stats: GenerationStats | None = None


@dataclass
class GenerationResult:
    text: str
    token_ids: list[int] = field(default_factory=list)
    stats: GenerationStats = field(default_factory=GenerationStats)


class _StopMatcher:
    """Holds back text that might be the start of a stop string."""

    def __init__(self, stops: list[str]):
        self.stops = [s for s in stops if s]
        self.buffer = ""

    def push(self, text: str) -> tuple[str, bool]:
        """Return (text safe to emit, stop_hit)."""
        self.buffer += text
        if not self.stops:
            out, self.buffer = self.buffer, ""
            return out, False
        hits = [i for i in (self.buffer.find(s) for s in self.stops) if i >= 0]
        if hits:
            out = self.buffer[: min(hits)]
            self.buffer = ""
            return out, True
        hold = 0
        for s in self.stops:
            for k in range(min(len(s) - 1, len(self.buffer)), 0, -1):
                if self.buffer.endswith(s[:k]):
                    hold = max(hold, k)
                    break
        cut = len(self.buffer) - hold
        out, self.buffer = self.buffer[:cut], self.buffer[cut:]
        return out, False

    def flush(self) -> str:
        out, self.buffer = self.buffer, ""
        return out


class InferenceEngine:
    """Owns a loaded model + tokenizer and generates text. Not thread-safe by itself:
    callers serialize access (the server wraps it in a lock)."""

    def __init__(self, loaded: LoadedCheckpoint, device: torch.device):
        self.ckpt = loaded
        self.model = loaded.model
        self.tokenizer = loaded.tokenizer
        self.device = device
        self.block_size = self.model.config.block_size
        self.device_reason = ""

    @classmethod
    def from_checkpoint(cls, path: str | Path, device: str = "auto") -> InferenceEngine:
        loaded = load_checkpoint(
            Path(path), "cpu"
        )  # validates the checkpoint before choosing a device
        info = select_inference_device(device, loaded.model.num_parameters())
        loaded.model.to(info.device)
        engine = cls(loaded, info.device)
        engine.device_reason = info.reason
        return engine

    @property
    def model_id(self) -> str:
        return checkpoint_id(self.ckpt.path)

    def describe(self) -> dict[str, Any]:
        return {
            "id": self.model_id,
            "device": str(self.device),
            "device_reason": self.device_reason,
            "num_parameters": self.model.num_parameters(),
            "model_config": self.model.config.__dict__,
            "tokenizer": self.tokenizer.describe(),
            "step": self.ckpt.meta.get("step"),
        }

    def encode_prompt(self, prompt: str, add_bos: bool = True) -> tuple[list[int], bool]:
        ids = self.tokenizer.encode(prompt)
        if add_bos:
            # Documents are EOS-separated in training data, so EOS doubles as start-of-text.
            ids = [self.tokenizer.eos_id, *ids]
        limit = self.block_size - 1
        truncated = len(ids) > limit
        if truncated:
            ids = ids[-limit:]
        return ids, truncated

    @torch.inference_mode()
    def stream(
        self,
        prompt: str,
        cfg: GenerationConfig | None = None,
        cancel: threading.Event | None = None,
        use_cache: bool = True,
        add_bos: bool = True,
    ) -> Iterator[GenerationEvent]:
        cfg = (cfg or GenerationConfig()).validate(self.tokenizer.vocab_size)
        t0 = time.perf_counter()
        ids, truncated = self.encode_prompt(prompt, add_bos=add_bos)
        stats = GenerationStats(prompt_tokens=len(ids), prompt_truncated=truncated)
        generator = None
        if not cfg.greedy:
            generator = torch.Generator()
            seed = (
                cfg.seed if cfg.seed is not None else int.from_bytes(os.urandom(8), "big") % 2**63
            )
            generator.manual_seed(seed)
        decoder = self.tokenizer.stream_decoder()
        matcher = _StopMatcher(cfg.stop)
        context = list(ids)
        cache = self.model.new_cache() if use_cache else None
        pending = torch.tensor([context], dtype=torch.long, device=self.device)
        t_first: float | None = None
        finish = "length"

        for _ in range(cfg.max_new_tokens):
            if cancel is not None and cancel.is_set():
                finish = "cancelled"
                break
            if use_cache:
                assert cache is not None
                if cache[0].length + pending.size(1) > self.block_size:
                    # Context window full: re-prefill from the most recent tokens.
                    keep = max(1, self.block_size // 2)
                    cache = self.model.new_cache()
                    pending = torch.tensor([context[-keep:]], dtype=torch.long, device=self.device)
                    stats.cache_rebuilds += 1
                logits, _ = self.model(pending, cache=cache, last_only=True)
            else:
                window = torch.tensor(
                    [context[-self.block_size :]], dtype=torch.long, device=self.device
                )
                logits, _ = self.model(window, last_only=True)
            last = logits[0, -1].float().cpu()
            previous = torch.tensor(context[-self.block_size :], dtype=torch.long)
            token = select_token(last, cfg, previous, generator)
            if t_first is None:
                t_first = time.perf_counter()
                stats.time_to_first_token_ms = round((t_first - t0) * 1000, 2)
            context.append(token)
            pending = torch.tensor([[token]], dtype=torch.long, device=self.device)
            if token == self.tokenizer.eos_id and cfg.stop_at_eos:
                finish = "eos"
                break
            stats.generated_tokens += 1
            text, hit = matcher.push(decoder.push(token))
            if text:
                yield GenerationEvent("token", text=text, token_id=token)
            if hit:
                finish = "stop"
                break

        if finish != "stop":
            tail = matcher.push(decoder.flush())[0] + matcher.flush()
            if tail and finish != "cancelled":
                yield GenerationEvent("token", text=tail)
        t_end = time.perf_counter()
        stats.finish_reason = finish
        stats.total_ms = round((t_end - t0) * 1000, 2)
        if t_first is not None and stats.generated_tokens > 1:
            stats.decode_tokens_per_sec = round(
                (stats.generated_tokens - 1) / max(1e-9, t_end - t_first), 2
            )
        if stats.generated_tokens:
            stats.tokens_per_sec = round(stats.generated_tokens / max(1e-9, t_end - t0), 2)
        yield GenerationEvent("done", stats=stats)

    def generate(
        self, prompt: str, cfg: GenerationConfig | None = None, **kwargs
    ) -> GenerationResult:
        parts: list[str] = []
        token_ids: list[int] = []
        stats = GenerationStats()
        for event in self.stream(prompt, cfg, **kwargs):
            if event.kind == "token":
                parts.append(event.text)
                if event.token_id is not None:
                    token_ids.append(event.token_id)
            elif event.stats is not None:
                stats = event.stats
        return GenerationResult("".join(parts), token_ids, stats)

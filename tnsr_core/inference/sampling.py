"""Next-token selection: greedy, temperature, top-k, top-p, repetition penalty."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import torch

MAX_NEW_TOKENS_LIMIT = 4096


class GenerationConfigError(ValueError):
    pass


@dataclass
class GenerationConfig:
    max_new_tokens: int = 200
    temperature: float = 0.8
    top_k: int = 0  # 0 disables top-k filtering
    top_p: float = 1.0  # 1.0 disables nucleus filtering
    repetition_penalty: float = 1.0  # 1.0 disables
    deterministic: bool = False  # greedy decoding (argmax); ignores temperature/top-k/top-p
    seed: int | None = None  # seeds sampling for reproducible stochastic output
    stop: list[str] = field(default_factory=list)  # stop strings (excluded from output)
    stop_at_eos: bool = True

    def validate(self, vocab_size: int | None = None) -> GenerationConfig:
        if not 1 <= self.max_new_tokens <= MAX_NEW_TOKENS_LIMIT:
            raise GenerationConfigError(f"max_new_tokens must be in [1, {MAX_NEW_TOKENS_LIMIT}]")
        if not 0.0 <= self.temperature <= 5.0:
            raise GenerationConfigError("temperature must be in [0, 5]")
        if self.top_k < 0 or (vocab_size is not None and self.top_k > vocab_size):
            raise GenerationConfigError(f"top_k must be in [0, {vocab_size or 'vocab_size'}]")
        if not 0.0 < self.top_p <= 1.0:
            raise GenerationConfigError("top_p must be in (0, 1]")
        if not 1.0 <= self.repetition_penalty <= 3.0:
            raise GenerationConfigError("repetition_penalty must be in [1, 3]")
        if self.seed is not None and not 0 <= self.seed < 2**63:
            raise GenerationConfigError("seed must be a non-negative 63-bit integer")
        if len(self.stop) > 8 or any(not s or len(s) > 64 for s in self.stop):
            raise GenerationConfigError("stop: at most 8 non-empty strings of <= 64 characters")
        return self

    @property
    def greedy(self) -> bool:
        return self.deterministic or self.temperature == 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def apply_repetition_penalty(
    logits: torch.Tensor, previous: torch.Tensor, penalty: float
) -> torch.Tensor:
    """CTRL-style penalty on tokens already present in the context. logits: (V,)."""
    if penalty == 1.0 or previous.numel() == 0:
        return logits
    logits = logits.clone()
    ids = torch.unique(previous)
    scores = logits[ids]
    logits[ids] = torch.where(scores > 0, scores / penalty, scores * penalty)
    return logits


def filter_logits(logits: torch.Tensor, top_k: int, top_p: float) -> torch.Tensor:
    """Mask logits outside the top-k / nucleus sets with -inf. logits: (V,)."""
    if top_k and top_k < logits.size(-1):
        kth = torch.topk(logits, top_k).values[-1]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(probs, dim=-1)
        # Drop tokens once the mass *before* them already reaches top_p (always keeps the top one).
        remove = (cumulative - probs) >= top_p
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(0, sorted_idx, sorted_logits)
    return logits


def select_token(
    logits: torch.Tensor,
    cfg: GenerationConfig,
    previous: torch.Tensor,
    generator: torch.Generator | None,
) -> int:
    """Pick the next token id from 1-D ``logits`` (float32, on CPU)."""
    logits = apply_repetition_penalty(logits, previous, cfg.repetition_penalty)
    if cfg.greedy:
        return int(torch.argmax(logits).item())
    logits = filter_logits(logits / cfg.temperature, cfg.top_k, cfg.top_p)
    probs = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, 1, generator=generator).item())

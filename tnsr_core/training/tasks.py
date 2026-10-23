"""Training tasks: what data the Trainer samples and how a model is evaluated.

``lm``    next-token language modeling on a prepared text dataset (``TokenDataset``)
``chess`` action-value prediction on a labeled chess dataset (``tnsr_core.chess``)
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

import numpy as np
import torch

from tnsr_core.config import RunConfig
from tnsr_core.tokenizer import Tokenizer


class Task(Protocol):
    kind: str

    @property
    def tokenizer(self) -> Tokenizer: ...

    @property
    def identity(self) -> str: ...

    def dataset_ref(self) -> dict[str, Any]: ...

    def train_batch(
        self, batch_size: int, rng: np.random.Generator
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    def eval_batches(
        self, split: str, n_batches: int, batch_size: int
    ) -> list[tuple[torch.Tensor, torch.Tensor]]: ...

    def tokens_per_example(self) -> int: ...

    def extra_eval(self, model: torch.nn.Module, device: torch.device) -> dict[str, float]: ...

    def checkpoint_meta(self) -> dict[str, Any]: ...


class LMTask:
    kind = "lm"

    def __init__(self, config: RunConfig, log: Callable[[str], None]):
        from tnsr_core.data import TokenDataset

        self.dataset = TokenDataset(config.data.dataset)
        self.tokenizer = self.dataset.tokenizer
        mc = config.model
        if mc.vocab_size != self.tokenizer.vocab_size:
            # The tokenizer is authoritative: align the model's vocabulary to it.
            log(
                f"Setting model.vocab_size={self.tokenizer.vocab_size} to match tokenizer {self.tokenizer.identity}"
            )
            mc.vocab_size = self.tokenizer.vocab_size
        self.dataset.check_block_size(mc.block_size)
        self.block_size = mc.block_size

    @property
    def identity(self) -> str:
        return self.dataset.identity

    def dataset_ref(self) -> dict[str, Any]:
        stats = self.dataset.meta["stats"]
        return {
            "name": self.dataset.name,
            "identity": self.dataset.identity,
            "val_text_sha256": self.dataset.meta.get("val_text_sha256"),
            "train_tokens": stats["train_tokens"],
            "val_tokens": stats["val_tokens"],
            "val_bytes": stats["val_bytes"],
        }

    def train_batch(
        self, batch_size: int, rng: np.random.Generator
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.dataset.random_batch("train", batch_size, self.block_size, rng)

    def eval_batches(
        self, split: str, n_batches: int, batch_size: int
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        return self.dataset.fixed_batches(split, n_batches, batch_size, self.block_size)

    def tokens_per_example(self) -> int:
        return self.block_size

    def extra_eval(self, model: torch.nn.Module, device: torch.device) -> dict[str, float]:
        return {}

    def checkpoint_meta(self) -> dict[str, Any]:
        return {}


def make_task(config: RunConfig, log: Callable[[str], None]) -> Task:
    if config.task == "chess":
        from tnsr_core.chess.task import ChessTask

        return ChessTask(config, log)
    return LMTask(config, log)

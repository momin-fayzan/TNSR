"""Memory-mapped token datasets and batch sampling."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tnsr_core.paths import datasets_dir, safe_join, validate_name
from tnsr_core.tokenizer import Tokenizer, load_tokenizer

from .prepare import DATASET_FORMAT_VERSION, DatasetError


def resolve_dataset_path(name_or_path: str) -> Path:
    path = Path(name_or_path)
    if path.is_dir() and (path / "meta.json").exists():
        return path.resolve()
    validate_name(name_or_path, "dataset name")
    path = safe_join(datasets_dir(), name_or_path)
    if not (path / "meta.json").is_file():
        raise DatasetError(
            f"Dataset {name_or_path!r} not found in {datasets_dir()}. "
            "Create it with: tnsr data prepare --name <name>"
        )
    return path


def read_dataset_meta(path: Path) -> dict[str, Any]:
    meta = json.loads((path / "meta.json").read_text())
    if meta.get("format_version") != DATASET_FORMAT_VERSION:
        raise DatasetError(f"Unsupported dataset format version in {path}")
    return meta


class TokenDataset:
    def __init__(self, name_or_path: str):
        self.path = resolve_dataset_path(name_or_path)
        self.meta = read_dataset_meta(self.path)
        self.tokenizer: Tokenizer = load_tokenizer(self.path / "tokenizer.json")
        if self.tokenizer.identity != self.meta["tokenizer"]["identity"]:
            raise DatasetError(f"Tokenizer in {self.path} does not match dataset metadata")
        dtype = np.dtype(self.meta["dtype"])
        self.splits = {
            split: np.memmap(self.path / f"{split}.bin", dtype=dtype, mode="r")
            for split in ("train", "val")
        }

    @property
    def name(self) -> str:
        return self.meta["name"]

    @property
    def identity(self) -> str:
        return self.meta["identity"]

    def num_tokens(self, split: str) -> int:
        return int(self.splits[split].size)

    def check_block_size(self, block_size: int) -> None:
        for split, data in self.splits.items():
            if data.size < block_size + 2:
                raise DatasetError(
                    f"The {split} split has only {data.size} tokens; need more than block_size+1 "
                    f"({block_size + 1}). Use a smaller block_size or a larger corpus."
                )

    def _windows(
        self, split: str, starts: np.ndarray, block_size: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        data = self.splits[split]
        idx = starts[:, None] + np.arange(block_size + 1)[None, :]
        chunk = torch.from_numpy(data[idx].astype(np.int64))
        return chunk[:, :-1], chunk[:, 1:]

    def random_batch(
        self, split: str, batch_size: int, block_size: int, rng: np.random.Generator
    ) -> tuple[torch.Tensor, torch.Tensor]:
        high = self.num_tokens(split) - block_size - 1
        starts = rng.integers(0, high + 1, size=batch_size)
        return self._windows(split, starts, block_size)

    def fixed_batches(
        self, split: str, n_batches: int, batch_size: int, block_size: int
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Deterministic evenly spaced windows, so losses are comparable across steps."""
        high = self.num_tokens(split) - block_size - 1
        total = n_batches * batch_size
        starts = np.linspace(0, high, num=total, dtype=np.int64)
        return [
            self._windows(split, starts[i * batch_size : (i + 1) * batch_size], block_size)
            for i in range(n_batches)
        ]

    def sequential_batches(
        self, split: str, block_size: int, batch_size: int, max_tokens: int | None = None
    ):
        """Non-overlapping windows covering the split (each target token scored exactly once)."""
        data = self.splits[split]
        n = data.size - 1
        if max_tokens is not None:
            n = min(n, max_tokens)
        n_windows = n // block_size
        starts = np.arange(n_windows, dtype=np.int64) * block_size
        for i in range(0, n_windows, batch_size):
            yield self._windows(split, starts[i : i + batch_size], block_size)

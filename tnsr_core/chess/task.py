"""Chess action-value training task: (position, move) -> win-probability bin."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import numpy as np
import torch

from tnsr_core.config import RunConfig

from .data import ChessSplit, iter_positions, resolve_chess_dataset
from .encoding import SEQ_LEN, ChessTokenizer, bin_centers

DEFAULT_BINS = 128
MOVE_EVAL_POSITIONS = 400


class ChessTask:
    kind = "chess"

    def __init__(self, config: RunConfig, log: Callable[[str], None]):
        self.path = resolve_chess_dataset(config.data.dataset)
        self.meta = json.loads((self.path / "meta.json").read_text())
        self.tokenizer = ChessTokenizer()
        mc = config.model
        fixed = {
            "vocab_size": self.tokenizer.vocab_size,
            "block_size": SEQ_LEN,
            "causal": False,
            "num_classes": mc.num_classes or DEFAULT_BINS,
        }
        for key, value in fixed.items():
            if getattr(mc, key) != value:
                log(f"chess task: setting model.{key}={value}")
                setattr(mc, key, value)
        self.num_bins = mc.num_classes
        self.train = ChessSplit(self.path, "train", self.tokenizer)
        self.val = ChessSplit(self.path, "val", self.tokenizer)
        self._eval_positions = list(iter_positions(self.val))[:MOVE_EVAL_POSITIONS]
        log(
            f"chess dataset {self.meta['name']}: {self.train.num_records:,} train / "
            f"{self.val.num_records:,} val (position, move) records"
        )

    @property
    def identity(self) -> str:
        return self.meta["identity"]

    def dataset_ref(self) -> dict[str, Any]:
        return {
            "name": self.meta["name"],
            "identity": self.meta["identity"],
            "kind": "chess_action_value",
            **self.meta.get("stats", {}),
            "labeling": self.meta.get("labeling"),
            "source": self.meta.get("source"),
        }

    def train_batch(
        self, batch_size: int, rng: np.random.Generator
    ) -> tuple[torch.Tensor, torch.Tensor]:
        idx = torch.from_numpy(rng.integers(0, self.train.num_records, size=batch_size))
        x, y, _ = self.train.batch(idx, self.num_bins)
        return x, y

    def eval_batches(
        self, split: str, n_batches: int, batch_size: int
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        data = self.train if split == "train" else self.val
        total = min(data.num_records, n_batches * batch_size)
        idx = torch.from_numpy(np.linspace(0, data.num_records - 1, num=total).astype(np.int64))
        out = []
        for i in range(0, total, batch_size):
            x, y, _ = data.batch(idx[i : i + batch_size], self.num_bins)
            out.append((x, y))
        return out

    def tokens_per_example(self) -> int:
        return SEQ_LEN

    @torch.no_grad()
    def extra_eval(self, model: torch.nn.Module, device: torch.device) -> dict[str, float]:
        """Move quality on held-out positions, judged against the Stockfish labels.

        move_accuracy: share of positions where the model's top move is Stockfish's top move.
        regret: mean drop in Stockfish win probability between the best move and the model's.
        """
        return move_quality(model, self.tokenizer, self._eval_positions, self.num_bins, device)


@torch.no_grad()
def move_quality(
    model: torch.nn.Module,
    tokenizer: ChessTokenizer,
    positions: list[tuple[str, list[tuple[str, float]]]],
    num_bins: int,
    device: torch.device,
) -> dict[str, float]:
    if not positions:
        return {}
    was_training = model.training
    model.eval()
    centers = torch.tensor(bin_centers(num_bins), dtype=torch.float32, device=device)
    hits = 0
    regrets = []
    for fen, moves in positions:
        state = torch.from_numpy(tokenizer.encode_fen(fen))
        x = state.unsqueeze(0).repeat(len(moves), 1)
        acts = torch.tensor([tokenizer.action_id(u) for u, _ in moves]).unsqueeze(1)
        logits, _ = model(torch.cat([x, acts], dim=1).to(device))
        expected = torch.softmax(logits.float(), dim=-1) @ centers
        choice = int(torch.argmax(expected))
        values = [v for _, v in moves]
        best = max(values)
        hits += int(values[choice] >= best - 1e-9)
        regrets.append(best - values[choice])
    if was_training:
        model.train()
    return {"move_accuracy": hits / len(positions), "regret": float(np.mean(regrets))}

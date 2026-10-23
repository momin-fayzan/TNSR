"""Chess action-value training task: (position, move) -> win-probability bin."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import numpy as np
import torch

from tnsr_core.config import RunConfig

from .data import ChessSplit, iter_positions, resolve_chess_dataset
from .encoding import SEQ_LEN, ChessTokenizer, ValueHead, encode_pairs

MOVE_EVAL_POSITIONS = 400


class ChessTask:
    kind = "chess"

    def __init__(self, config: RunConfig, log: Callable[[str], None]):
        # "a+b" trains on several datasets at once (e.g. Lichess positions + on-policy games).
        self.paths = [resolve_chess_dataset(n) for n in config.data.dataset.split("+")]
        self.metas = [json.loads((p / "meta.json").read_text()) for p in self.paths]
        self.meta = self.metas[0]
        self.tokenizer = ChessTokenizer(move_marks=config.data.move_marks)
        mc = config.model
        self.head = ValueHead(config.data.value_bins, config.data.mate_classes)
        fixed = {
            "vocab_size": self.tokenizer.vocab_size,
            "block_size": SEQ_LEN,
            "causal": False,
            "num_classes": self.head.num_classes,
        }
        for key, value in fixed.items():
            if getattr(mc, key) != value:
                log(f"chess task: setting model.{key}={value}")
                setattr(mc, key, value)
        self.value_loss = config.train.value_loss
        self.train = ChessSplit(self.paths, "train", self.tokenizer)
        self.val = ChessSplit(self.paths, "val", self.tokenizer)
        # Move quality is always measured on the first dataset's validation positions, so runs
        # that add extra datasets stay comparable with runs that do not.
        self._eval_positions = list(iter_positions(self.val))[:MOVE_EVAL_POSITIONS]
        log(
            f"chess dataset {self.name}: {self.train.num_records:,} train / "
            f"{self.val.num_records:,} val (position, move) records"
        )

    @property
    def name(self) -> str:
        return "+".join(m["name"] for m in self.metas)

    @property
    def identity(self) -> str:
        return "+".join(m["identity"] for m in self.metas)

    def dataset_ref(self) -> dict[str, Any]:
        stats: dict[str, int] = {}
        for m in self.metas:
            for k, v in m.get("stats", {}).items():
                stats[k] = stats.get(k, 0) + v
        ref = {
            "name": self.name,
            "identity": self.identity,
            "kind": "chess_action_value",
            **stats,
            "labeling": self.meta.get("labeling"),
            "source": self.meta.get("source"),
        }
        if len(self.metas) > 1:
            ref["sources"] = [m.get("source") for m in self.metas]
        return ref

    def train_batch(
        self, batch_size: int, rng: np.random.Generator
    ) -> tuple[torch.Tensor, torch.Tensor]:
        idx = torch.from_numpy(rng.integers(0, self.train.num_records, size=batch_size))
        x, y, _ = self.train.batch(idx, self.head)
        if self.value_loss == "hl_gauss":
            # Soft targets for training; evaluation keeps hard classes so losses stay comparable.
            return x, self.train.soft_targets(idx, self.head)
        return x, y

    def eval_batches(
        self, split: str, n_batches: int, batch_size: int
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        data = self.train if split == "train" else self.val
        total = min(data.num_records, n_batches * batch_size)
        idx = torch.from_numpy(np.linspace(0, data.num_records - 1, num=total).astype(np.int64))
        out = []
        for i in range(0, total, batch_size):
            x, y, _ = data.batch(idx[i : i + batch_size], self.head)
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
        return move_quality(model, self.tokenizer, self._eval_positions, self.head, device)

    def checkpoint_meta(self) -> dict[str, Any]:
        return {"chess": self.head.to_dict()}


@torch.no_grad()
def move_quality(
    model: torch.nn.Module,
    tokenizer: ChessTokenizer,
    positions: list[tuple[str, list[tuple[str, float]]]],
    head: ValueHead | int,
    device: torch.device,
) -> dict[str, float]:
    if not positions:
        return {}
    if isinstance(head, int):
        head = ValueHead(head)
    was_training = model.training
    model.eval()
    rank = torch.tensor(head.rank_values, dtype=torch.float32, device=device)
    hits = 0
    regrets = []
    for fen, moves in positions:
        x = torch.from_numpy(encode_pairs(tokenizer, fen, [u for u, _ in moves]))
        logits, _ = model(x.to(device))
        choice = int(torch.argmax(torch.softmax(logits.float(), dim=-1) @ rank))
        values = [v for _, v in moves]
        best = max(values)
        hits += int(values[choice] >= best - 1e-9)
        regrets.append(best - values[choice])
    if was_training:
        model.train()
    return {"move_accuracy": hits / len(positions), "regret": float(np.mean(regrets))}

"""Chess action-value training task: (position, move) -> win-probability bin."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import numpy as np
import torch

from tnsr_core.config import RunConfig

from .data import ChessSplit, iter_positions, resolve_chess_dataset
from .encoding import SEQ_LEN, STATE_LEN, ChessTokenizer, ValueHead, move_logits

MOVE_EVAL_POSITIONS = 400


class ChessTask:
    kind = "chess"

    def __init__(self, config: RunConfig, log: Callable[[str], None]):
        # "a+b" trains on several datasets at once (e.g. Lichess positions + on-policy games).
        self.paths = [resolve_chess_dataset(n) for n in config.data.dataset.split("+")]
        self.metas = [json.loads((p / "meta.json").read_text()) for p in self.paths]
        self.meta = self.metas[0]
        # Shared board: one sequence per position (board + all its moves) and batch_size counts
        # positions; otherwise one sequence per (position, move) record.
        self.shared = config.data.shared_board
        if self.shared and config.data.move_marks:
            raise ValueError("data.shared_board needs data.move_marks = false")
        self.tokenizer = ChessTokenizer(move_marks=config.data.move_marks)
        mc = config.model
        self.head = ValueHead(config.data.value_bins, config.data.mate_classes)
        fixed = {
            "vocab_size": self.tokenizer.vocab_size,
            "block_size": SEQ_LEN,
            "causal": False,
            "num_classes": self.head.num_classes,
            "query_prefix": STATE_LEN if self.shared else 0,
        }
        for key, value in fixed.items():
            if getattr(mc, key) != value:
                log(f"chess task: setting model.{key}={value}")
                setattr(mc, key, value)
        self.value_loss = config.train.value_loss
        self.weights = _parse_weights(config.data.dataset_weights, len(self.paths))
        if self.weights is None:
            self.parts = [ChessSplit(self.paths, "train", self.tokenizer, load_fens=False)]
        else:  # one split per dataset, so each batch can take a fixed share from each
            self.parts = [
                ChessSplit(p, "train", self.tokenizer, load_fens=False) for p in self.paths
            ]
        # Training-loss estimates use the first part (all data unless weights are set).
        self.train = self.parts[0]
        self.val = ChessSplit(self.paths, "val", self.tokenizer)
        self._part_positions = [self._labeled_positions(p) for p in self.parts]
        # Move quality is always measured on the first dataset's validation positions, so runs
        # that add extra datasets stay comparable with runs that do not.
        self._eval_positions = list(iter_positions(self.val))[:MOVE_EVAL_POSITIONS]
        n_train = sum(p.num_records for p in self.parts)
        log(
            f"chess dataset {self.name}: {n_train:,} train / "
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
        if self.weights is not None:
            ref["weights"] = [round(float(w), 4) for w in self.weights]
        return ref

    @staticmethod
    def _labeled_positions(split: ChessSplit) -> torch.Tensor:
        return torch.nonzero(split.pos_offsets[1:] > split.pos_offsets[:-1]).squeeze(1)

    def train_batch(
        self, batch_size: int, rng: np.random.Generator
    ) -> tuple[torch.Tensor, torch.Tensor]:
        counts = [batch_size] if self.weights is None else rng.multinomial(batch_size, self.weights)
        batches = [
            self._part_batch(part, positions, int(n), rng)
            for part, positions, n in zip(self.parts, self._part_positions, counts, strict=True)
            if n
        ]
        if len(batches) == 1:
            return batches[0]
        return _concat_padded(batches)

    def _part_batch(
        self, part: ChessSplit, positions: torch.Tensor, n: int, rng: np.random.Generator
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.shared:
            pick = torch.from_numpy(rng.integers(0, len(positions), size=n))
            return part.position_batch(
                positions[pick], self.head, soft=self.value_loss == "hl_gauss"
            )
        idx = torch.from_numpy(rng.integers(0, part.num_records, size=n))
        x, y, _ = part.batch(idx, self.head)
        if self.value_loss == "hl_gauss":
            # Soft targets for training; evaluation keeps hard classes so losses stay comparable.
            return x, part.soft_targets(idx, self.head)
        return x, y

    def eval_batches(
        self, split: str, n_batches: int, batch_size: int
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        data = self.train if split == "train" else self.val
        if self.shared:
            pos = self._labeled_positions(data)
            n = min(len(pos), n_batches * batch_size)
            pick = pos[np.linspace(0, len(pos) - 1, num=n).astype(np.int64)]
            return [
                data.position_batch(pick[i : i + batch_size], self.head)
                for i in range(0, n, batch_size)
            ]
        total = min(data.num_records, n_batches * batch_size)
        idx = torch.from_numpy(np.linspace(0, data.num_records - 1, num=total).astype(np.int64))
        out = []
        for i in range(0, total, batch_size):
            x, y, _ = data.batch(idx[i : i + batch_size], self.head)
            out.append((x, y))
        return out

    def tokens_per_example(self) -> int:
        if self.shared:  # board + average number of labeled moves
            records = sum(p.num_records for p in self.parts)
            positions = sum(p.num_positions for p in self.parts)
            return int(round(STATE_LEN + records / max(1, positions)))
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


def _parse_weights(text: str, n: int) -> np.ndarray | None:
    if not text.strip():
        return None
    try:
        weights = np.array([float(w) for w in text.split(",")])
    except ValueError as exc:
        raise ValueError(f"data.dataset_weights must be numbers like '1,2': {text!r}") from exc
    if len(weights) != n or (weights < 0).any() or weights.sum() <= 0:
        raise ValueError(
            f"data.dataset_weights needs {n} non-negative weights (one per dataset), got {text!r}"
        )
    return weights / weights.sum()


def _concat_padded(
    batches: list[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Concatenate batches whose sequences may differ in length (shared-board batches)."""
    width = max(x.size(1) for x, _ in batches)
    xs, ys = [], []
    for x, y in batches:
        pad = width - x.size(1)
        if pad:
            x = torch.nn.functional.pad(x, (0, pad))
            if y.is_floating_point():
                y = torch.nn.functional.pad(y, (0, 0, 0, pad))
            else:
                y = torch.nn.functional.pad(y, (0, pad), value=-100)
        xs.append(x)
        ys.append(y)
    return torch.cat(xs), torch.cat(ys)


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
        logits = move_logits(model, tokenizer, fen, [u for u, _ in moves], device)
        choice = int(torch.argmax(torch.softmax(logits.float(), dim=-1) @ rank))
        values = [v for _, v in moves]
        best = max(values)
        hits += int(values[choice] >= best - 1e-9)
        regrets.append(best - values[choice])
    if was_training:
        model.train()
    return {"move_accuracy": hits / len(positions), "regret": float(np.mean(regrets))}

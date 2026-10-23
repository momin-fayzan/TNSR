"""TNSR chess engine: one batched forward pass over the legal moves, no search.

For a position, every legal move is paired with the position encoding, the model predicts a
win-probability distribution for each pair, and the engine plays the move with the highest
expected win probability. Illegal moves are impossible because only legal moves are scored.

With ``rules=True`` the engine also applies the game rules to the move itself: a move that
checkmates is worth a certain win and a move that ends the game in a draw (stalemate,
insufficient material, a repetition or fifty-move claim) is worth exactly one half. This looks
at no opponent replies and calls no model on any later position; it replaces the model's
estimate only where the rules already know the answer, including repetitions, which the
model cannot see because a FEN has no move history.
"""

from __future__ import annotations

import math
import sys
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

import chess
import torch

from tnsr_core.checkpoints import CheckpointError, checkpoint_id, load_checkpoint
from tnsr_core.device import select_inference_device

from .encoding import ChessTokenizer, ValueHead, encode_pairs


@dataclass
class MoveScore:
    uci: str
    san: str
    win_prob: float  # expected win probability for the side to move after this move


class ChessEngine:
    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: ChessTokenizer,
        device: torch.device,
        head: ValueHead | int,
        model_id: str = "",
        rules: bool = False,
    ):
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.device = device
        self.head = ValueHead(head) if isinstance(head, int) else head
        self.model_id = model_id
        self.rules = rules
        self._win = torch.tensor(self.head.win_values, dtype=torch.float32, device=device)
        self._rank = torch.tensor(self.head.rank_values, dtype=torch.float32, device=device)

    @property
    def num_bins(self) -> int:
        return self.head.value_bins

    @classmethod
    def from_checkpoint(
        cls, path: str | Path, device: str = "auto", rules: bool = False
    ) -> ChessEngine:
        loaded = load_checkpoint(Path(path), "cpu")
        if not isinstance(loaded.tokenizer, ChessTokenizer) or loaded.meta.get("task") != "chess":
            raise CheckpointError(f"{path} is not a TNSR chess checkpoint")
        try:
            head = ValueHead.from_meta(loaded.meta, loaded.config.num_classes)
        except ValueError as exc:
            raise CheckpointError(f"{path}: {exc}") from exc
        info = select_inference_device(device, loaded.model.num_parameters())
        loaded.model.to(info.device)
        return cls(
            loaded.model, loaded.tokenizer, info.device, head, checkpoint_id(Path(path)), rules
        )

    @torch.no_grad()
    def analyse(self, board: chess.Board) -> list[MoveScore]:
        """All legal moves, best first (ranked by expected win probability, faster mates first)."""
        moves = list(board.legal_moves)
        if not moves:
            return []
        x = torch.from_numpy(
            encode_pairs(self.tokenizer, board.fen(), [m.uci() for m in moves])
        ).to(self.device)
        logits, _ = self.model(x)
        probs = torch.softmax(logits.float(), dim=-1)
        win = (probs @ self._win).cpu().tolist()
        rank = (probs @ self._rank).cpu().tolist()
        if self.rules:
            for i, m in enumerate(moves):
                result = move_result(board, m)
                if result is not None:
                    win[i] = result
                    rank[i] = 2.0 if result == 1.0 else result  # mate in one beats everything
        scored = [
            (r, MoveScore(m.uci(), board.san(m), w))
            for m, w, r in zip(moves, win, rank, strict=True)
        ]
        scored.sort(key=lambda t: t[0], reverse=True)
        return [s for _, s in scored]

    def best_move(self, board: chess.Board) -> chess.Move:
        scores = self.analyse(board)
        if not scores:
            raise ValueError("No legal moves: the game is over")
        return chess.Move.from_uci(scores[0].uci)


def move_result(board: chess.Board, move: chess.Move) -> float | None:
    """Exact result for the mover if ``move`` ends the game by rule (1 or 0.5), else None."""
    board.push(move)
    try:
        if board.is_checkmate():
            return 1.0
        if board.is_stalemate() or board.is_insufficient_material() or board.can_claim_draw():
            return 0.5
        return None
    finally:
        board.pop()


def win_prob_to_cp(p: float) -> int:
    """Inverse of the Lichess win-probability model, clamped to +-2000 centipawns."""
    p = min(max(p, 1e-4), 1 - 1e-4)
    return int(max(-2000, min(2000, round(-math.log(1 / p - 1) / 0.00368208))))


# -- UCI protocol ----------------------------------------------------------------------------


def _parse_position(tokens: list[str]) -> chess.Board:
    if not tokens:
        return chess.Board()
    if tokens[0] == "startpos":
        board = chess.Board()
        rest = tokens[1:]
    elif tokens[0] == "fen":
        fen_parts = []
        rest = tokens[1:]
        while rest and rest[0] != "moves":
            fen_parts.append(rest.pop(0))
        board = chess.Board(" ".join(fen_parts))
    else:
        raise ValueError(f"bad position command: {' '.join(tokens)}")
    if rest and rest[0] == "moves":
        for uci in rest[1:]:
            board.push_uci(uci)
    return board


def run_uci(
    engine_factory: Callable[[], ChessEngine],
    stdin: Iterable[str] = sys.stdin,
    stdout: TextIO = sys.stdout,
) -> None:
    """Minimal UCI loop (uci, isready, ucinewgame, position, go, stop, quit)."""

    def say(line: str) -> None:
        stdout.write(line + "\n")
        stdout.flush()

    engine: ChessEngine | None = None
    board = chess.Board()
    for raw in stdin:
        parts = raw.strip().split()
        if not parts:
            continue
        cmd, args = parts[0], parts[1:]
        if cmd == "uci":
            say("id name TNSR (transformer, no search)")
            say("id author TNSR")
            say("uciok")
        elif cmd == "isready":
            if engine is None:
                engine = engine_factory()
            say("readyok")
        elif cmd == "ucinewgame":
            board = chess.Board()
        elif cmd == "position":
            try:
                board = _parse_position(args)
            except ValueError as exc:
                say(f"info string error: {exc}")
        elif cmd == "go":
            if engine is None:
                engine = engine_factory()
            t0 = time.perf_counter()
            scores = engine.analyse(board)
            if not scores:
                say("bestmove 0000")
                continue
            best = scores[0]
            cp = win_prob_to_cp(best.win_prob)
            ms = int((time.perf_counter() - t0) * 1000)
            say(
                f"info depth 1 score cp {cp} time {ms} pv {best.uci} string win_prob {best.win_prob:.3f}"
            )
            say(f"bestmove {best.uci}")
        elif cmd == "quit":
            break

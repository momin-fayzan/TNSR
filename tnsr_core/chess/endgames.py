"""Random endgame positions for teaching the engine to convert won positions.

Positions sampled from human games rarely contain the lopsided endings an engine reaches once
it is winning (e.g. king and queen against a bare king), so a model trained only on them wins
material and then shuffles or stalemates. These generated positions are labeled by Stockfish
exactly like game positions; nothing about play-time inference changes.
"""

from __future__ import annotations

import random

import chess

# (stronger side's extra pieces, weaker side's extra pieces); kings are always added.
MATERIAL = [
    ("Q", ""), ("R", ""), ("QR", ""), ("RR", ""), ("BB", ""), ("BN", ""),
    ("Q", "R"), ("Q", "B"), ("Q", "N"), ("R", "B"), ("R", "N"), ("Q", "P"),
    ("P", ""), ("PP", ""), ("RP", "R"), ("QP", "Q"), ("RP", ""), ("NP", ""),
    ("BP", ""), ("PPP", "P"), ("RPP", "R"), ("Q", "RR"), ("QQ", "Q"), ("R", "P"),
]  # fmt: skip


def random_endgame(rng: random.Random, max_tries: int = 200) -> chess.Board | None:
    strong, weak = rng.choice(MATERIAL)
    strong_color = rng.choice([chess.WHITE, chess.BLACK])
    for _ in range(max_tries):
        board = chess.Board(None)
        squares = rng.sample(chess.SQUARES, 2 + len(strong) + len(weak))
        board.set_piece_at(squares[0], chess.Piece(chess.KING, strong_color))
        board.set_piece_at(squares[1], chess.Piece(chess.KING, not strong_color))
        ok = True
        for sq, sym in zip(squares[2:], list(strong) + list(weak), strict=True):
            color = strong_color if sq in squares[2 : 2 + len(strong)] else not strong_color
            if sym == "P" and chess.square_rank(sq) in (0, 7):
                ok = False
                break
            board.set_piece_at(
                sq, chess.Piece.from_symbol(sym if color == chess.WHITE else sym.lower())
            )
        if not ok:
            continue
        board.turn = rng.choice([chess.WHITE, chess.BLACK])
        board.halfmove_clock = rng.randint(0, 40)
        board.fullmove_number = rng.randint(30, 90)
        if board.is_valid() and not board.is_game_over():
            return board
    return None


def generate_endgames(n: int, seed: int = 0) -> list[str]:
    """``n`` distinct, legal, non-terminal endgame FENs (deterministic for a seed)."""
    rng = random.Random(seed)
    out: list[str] = []
    seen: set[str] = set()
    attempts = 0
    while len(out) < n and attempts < n * 20:
        attempts += 1
        board = random_endgame(rng)
        if board is None:
            continue
        key = board.epd()
        if key not in seen:
            seen.add(key)
            out.append(board.fen())
    return out

"""Fixed-length chess tokenization (positions and moves) and win-probability bins.

A position is encoded as exactly 77 characters, following the FEN fields:

    side to move (1) | board, 64 squares a8..h1 with '.' for empty (64) |
    castling rights padded to 4 | en-passant square padded to 2 |
    halfmove clock padded to 3 | fullmove number padded to 3

Moves use a fixed vocabulary of every UCI move that is geometrically possible on an
empty board (queen and knight moves from every square, plus promotions): 1,968 actions.
Sequences fed to the model are the 77 position tokens followed by one action token.
"""

from __future__ import annotations

import functools
from collections.abc import Iterable
from typing import Any

import chess
import numpy as np

from tnsr_core.tokenizer.base import EOS, Tokenizer

STATE_LEN = 77
SEQ_LEN = STATE_LEN + 1  # position + candidate move
CHARS = sorted(set("wb") | set("KQRBNPkqrbnp") | set(".-") | set("0123456789") | set("abcdefgh"))
CHESS_TOKENIZER_VERSION = 1


@functools.cache
def all_actions() -> tuple[str, ...]:
    """Every UCI move string that can occur in standard chess (1,968), in a fixed order."""
    moves: set[str] = set()
    for frm in chess.SQUARES:
        empty = chess.BaseBoard.empty()
        empty.set_piece_at(frm, chess.Piece(chess.QUEEN, chess.WHITE))
        for to in empty.attacks(frm):
            moves.add(chess.square_name(frm) + chess.square_name(to))
        empty = chess.BaseBoard.empty()
        empty.set_piece_at(frm, chess.Piece(chess.KNIGHT, chess.WHITE))
        for to in empty.attacks(frm):
            moves.add(chess.square_name(frm) + chess.square_name(to))
    for color_rank, target_rank in ((6, 7), (1, 0)):
        for file in range(8):
            frm = chess.square(file, color_rank)
            for df in (-1, 0, 1):
                if 0 <= file + df < 8:
                    to = chess.square(file + df, target_rank)
                    for piece in "qrbn":
                        moves.add(chess.square_name(frm) + chess.square_name(to) + piece)
    return tuple(sorted(moves))


class ChessTokenizer(Tokenizer):
    """Token ids: [EOS] + position characters + action vocabulary."""

    kind = "chess"

    def __init__(self) -> None:
        self.special_tokens = [EOS]
        self.chars = list(CHARS)
        self.actions = list(all_actions())
        self.char_offset = 1
        self.action_offset = self.char_offset + len(self.chars)
        self._char_ids = {c: i + self.char_offset for i, c in enumerate(self.chars)}
        self._action_ids = {a: i + self.action_offset for i, a in enumerate(self.actions)}
        # Byte -> id lookup table for fast vectorised position encoding.
        self._lut = np.zeros(256, dtype=np.int64)
        for c, i in self._char_ids.items():
            self._lut[ord(c)] = i

    @property
    def vocab_size(self) -> int:
        return self.action_offset + len(self.actions)

    @property
    def num_actions(self) -> int:
        return len(self.actions)

    # -- positions -------------------------------------------------------------------

    @staticmethod
    def fen_to_chars(fen: str) -> str:
        board, side, castling, ep, halfmove, fullmove = fen.split()
        squares = "".join("." * int(c) if c.isdigit() else c for c in board.replace("/", ""))
        if len(squares) != 64:
            raise ValueError(f"Invalid FEN board: {fen}")
        castling = castling.replace("-", "")
        return (
            side
            + squares
            + castling.ljust(4, ".")
            + ep.ljust(2, ".")
            + str(min(int(halfmove), 999)).ljust(3, ".")
            + str(min(int(fullmove), 999)).ljust(3, ".")
        )

    def encode_fen(self, fen: str) -> np.ndarray:
        chars = self.fen_to_chars(fen)
        assert len(chars) == STATE_LEN
        return self._lut[np.frombuffer(chars.encode("ascii"), dtype=np.uint8)]

    def encode(self, text: str, add_eos: bool = False) -> list[int]:
        ids = self.encode_fen(text).tolist()
        if add_eos:
            ids.append(self.eos_id)
        return ids

    # -- moves -----------------------------------------------------------------------

    def action_id(self, uci: str) -> int:
        try:
            return self._action_ids[uci]
        except KeyError as exc:
            raise ValueError(f"Not a chess move in the action vocabulary: {uci!r}") from exc

    def action_index(self, uci: str) -> int:
        """0-based index into the action vocabulary (for compact storage)."""
        return self.action_id(uci) - self.action_offset

    def action_from_index(self, index: int) -> str:
        return self.actions[index]

    def token_bytes(self, token_id: int) -> bytes:
        self._check_id(token_id)
        if token_id < self.char_offset:
            return b""
        if token_id < self.action_offset:
            return self.chars[token_id - self.char_offset].encode()
        return self.actions[token_id - self.action_offset].encode()

    def _payload(self) -> dict[str, Any]:
        return {
            "version": CHESS_TOKENIZER_VERSION,
            "chars": self.chars,
            "num_actions": self.num_actions,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ChessTokenizer:
        tok = cls()
        if raw.get("chars") != tok.chars or raw.get("num_actions") != tok.num_actions:
            raise ValueError("Chess tokenizer layout does not match this TNSR build")
        if raw.get("version") != CHESS_TOKENIZER_VERSION:
            raise ValueError(f"Unsupported chess tokenizer version {raw.get('version')!r}")
        return tok


# -- win-probability bins -----------------------------------------------------------------


def bin_edges(num_bins: int) -> np.ndarray:
    return np.linspace(0.0, 1.0, num_bins + 1)


def bin_centers(num_bins: int) -> np.ndarray:
    edges = bin_edges(num_bins)
    return (edges[:-1] + edges[1:]) / 2


def win_prob_to_bin(win_prob: np.ndarray | float, num_bins: int) -> np.ndarray:
    """Uniform bins over [0, 1]; 1.0 falls in the last bin."""
    p = np.clip(np.asarray(win_prob, dtype=np.float64), 0.0, 1.0)
    return np.minimum((p * num_bins).astype(np.int64), num_bins - 1)


def legal_ucis(board: chess.Board) -> list[str]:
    return [m.uci() for m in board.legal_moves]


def encode_pairs(tokenizer: ChessTokenizer, fen: str, ucis: Iterable[str]) -> np.ndarray:
    """(num_moves, SEQ_LEN) token ids: the position followed by each candidate move."""
    state = tokenizer.encode_fen(fen)
    ucis = list(ucis)
    out = np.empty((len(ucis), SEQ_LEN), dtype=np.int64)
    out[:, :STATE_LEN] = state
    out[:, STATE_LEN] = [tokenizer.action_id(u) for u in ucis]
    return out

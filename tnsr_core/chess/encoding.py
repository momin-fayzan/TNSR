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
SQUARE_CHARS = "KQRBNPkqrbnp."  # what can stand on a board square
CHESS_TOKENIZER_VERSION = 2  # v2 adds optional move marks; v1 (no marks) still loads


def board_position(square: str) -> int:
    """Index of a square inside the 77-token state (0 is the side-to-move token, a8 is 1)."""
    file = ord(square[0]) - ord("a")
    rank = int(square[1])
    return 1 + (8 - rank) * 8 + file


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
    """Token ids: [EOS] + position characters [+ move-marked square variants] + actions.

    With ``move_marks`` the candidate move is also written onto the board: the token on its
    from-square and its to-square are replaced by "from"/"to" variants, so attention can see
    which pieces and squares the move involves instead of decoding an opaque move id.
    """

    kind = "chess"

    def __init__(self, move_marks: bool = True) -> None:
        self.special_tokens = [EOS]
        self.chars = list(CHARS)
        self.actions = list(all_actions())
        self.move_marks = move_marks
        self.char_offset = 1
        self._char_ids = {c: i + self.char_offset for i, c in enumerate(self.chars)}
        next_id = self.char_offset + len(self.chars)
        if move_marks:
            self.from_offset = next_id
            self.to_offset = next_id + len(SQUARE_CHARS)
            next_id = self.to_offset + len(SQUARE_CHARS)
        self.action_offset = next_id
        self._action_ids = {a: i + self.action_offset for i, a in enumerate(self.actions)}
        # Byte -> id lookup table for fast vectorised position encoding.
        self._lut = np.zeros(256, dtype=np.int64)
        for c, i in self._char_ids.items():
            self._lut[ord(c)] = i
        if move_marks:
            self._from_lut = np.arange(self.action_offset, dtype=np.int64)
            self._to_lut = np.arange(self.action_offset, dtype=np.int64)
            for j, c in enumerate(SQUARE_CHARS):
                self._from_lut[self._char_ids[c]] = self.from_offset + j
                self._to_lut[self._char_ids[c]] = self.to_offset + j
            self._from_pos = np.array([board_position(a[:2]) for a in self.actions], dtype=np.int64)
            self._to_pos = np.array([board_position(a[2:4]) for a in self.actions], dtype=np.int64)

    def mark_moves(self, states: Any, action_index: Any) -> Any:
        """Return copies of (B, 77) state token ids with each row's move marked on the board.

        Accepts numpy arrays or torch tensors (returns the same type)."""
        import torch

        as_torch = isinstance(states, torch.Tensor)
        st = states.cpu().numpy().copy() if as_torch else np.array(states, copy=True)
        act = (
            action_index.cpu().numpy()
            if isinstance(action_index, torch.Tensor)
            else np.asarray(action_index)
        )
        rows = np.arange(st.shape[0])
        fp, tp = self._from_pos[act], self._to_pos[act]
        st[rows, fp] = self._from_lut[st[rows, fp]]
        st[rows, tp] = self._to_lut[st[rows, tp]]
        return torch.from_numpy(st) if as_torch else st

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
        if self.move_marks and self.from_offset <= token_id < self.action_offset:
            # a square marked as the move's from/to square: same character, different token
            return SQUARE_CHARS[(token_id - self.from_offset) % len(SQUARE_CHARS)].encode()
        if token_id < self.action_offset:
            return self.chars[token_id - self.char_offset].encode()
        return self.actions[token_id - self.action_offset].encode()

    def _payload(self) -> dict[str, Any]:
        if not self.move_marks:  # identical to the v1 layout, so v1 identities are preserved
            return {"version": 1, "chars": self.chars, "num_actions": self.num_actions}
        return {
            "version": CHESS_TOKENIZER_VERSION,
            "chars": self.chars,
            "num_actions": self.num_actions,
            "move_marks": True,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ChessTokenizer:
        version = raw.get("version")
        if version not in (1, 2):
            raise ValueError(f"Unsupported chess tokenizer version {version!r}")
        tok = cls(move_marks=bool(raw.get("move_marks", False)))
        if raw.get("chars") != tok.chars or raw.get("num_actions") != tok.num_actions:
            raise ValueError("Chess tokenizer layout does not match this TNSR build")
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


def hl_gauss_targets(win_prob: np.ndarray, num_bins: int, sigma_bins: float = 0.75) -> np.ndarray:
    """Histogram-loss Gaussian targets (Imani et al., 2024): the probability mass a Gaussian
    centred on each label puts in every bin. Shape (N, num_bins), rows sum to 1."""
    import torch

    edges = torch.from_numpy(bin_edges(num_bins))
    sigma = sigma_bins / num_bins
    p = torch.as_tensor(np.asarray(win_prob, dtype=np.float64)).unsqueeze(1)
    cdf = 0.5 * (1 + torch.erf((edges.unsqueeze(0) - p) / (sigma * 2**0.5)))
    mass = torch.diff(cdf, dim=1)
    return (mass / mass.sum(dim=1, keepdim=True)).numpy()


def legal_ucis(board: chess.Board) -> list[str]:
    return [m.uci() for m in board.legal_moves]


def encode_pairs(tokenizer: ChessTokenizer, fen: str, ucis: Iterable[str]) -> np.ndarray:
    """(num_moves, SEQ_LEN) token ids: the position followed by each candidate move."""
    state = tokenizer.encode_fen(fen)
    ucis = list(ucis)
    out = np.empty((len(ucis), SEQ_LEN), dtype=np.int64)
    out[:, :STATE_LEN] = state
    out[:, STATE_LEN] = [tokenizer.action_id(u) for u in ucis]
    if tokenizer.move_marks and ucis:
        out[:, :STATE_LEN] = tokenizer.mark_moves(
            out[:, :STATE_LEN], out[:, STATE_LEN] - tokenizer.action_offset
        )
    return out


# -- value head: win-probability bins + mate-distance classes ------------------------------

MATE_RANK_BONUS = 0.05  # ranking bonus for a certain fastest mate; below one bin of win prob


class ValueHead:
    """Output classes of the action-value model and how to read them.

    Classes ``0 .. value_bins-1`` are uniform win-probability bins. With ``mate_classes = M``
    the next M classes mean "the mover mates in n" (n = 1..M) and the last M "the mover gets
    mated in n". Mates further than M (or of unknown distance) use the n = M class.
    """

    def __init__(self, value_bins: int = 128, mate_classes: int = 0):
        if value_bins < 2 or mate_classes < 0:
            raise ValueError("value_bins must be >= 2 and mate_classes >= 0")
        self.value_bins = value_bins
        self.mate_classes = mate_classes
        m = mate_classes
        centers = bin_centers(value_bins)
        speed = (m + 1 - np.arange(1, m + 1)) / (m + 1) if m else np.zeros(0)  # 1.0 for mate in 1
        # Expected win probability for the mover (what the UI shows).
        self.win_values = np.concatenate([centers, np.ones(m), np.zeros(m)])
        # Ranking score: win probability plus a small preference for faster mates (and for
        # slower ones when getting mated), so a won game is actually converted.
        self.rank_values = self.win_values + np.concatenate(
            [np.zeros(value_bins), MATE_RANK_BONUS * speed, -MATE_RANK_BONUS * speed]
        )

    @property
    def num_classes(self) -> int:
        return self.value_bins + 2 * self.mate_classes

    def classes(self, win_prob: np.ndarray, mate: np.ndarray) -> np.ndarray:
        cls = win_prob_to_bin(win_prob, self.value_bins)
        if self.mate_classes:
            mate = np.asarray(mate)
            dist = np.minimum(np.abs(mate), self.mate_classes) - 1
            cls = np.where(mate > 0, self.value_bins + dist, cls)
            cls = np.where(mate < 0, self.value_bins + self.mate_classes + dist, cls)
        return cls

    def soft_targets(self, win_prob: np.ndarray, mate: np.ndarray) -> np.ndarray:
        """HL-Gauss over the win-probability bins; one-hot for mate classes."""
        out = np.zeros((len(win_prob), self.num_classes))
        out[:, : self.value_bins] = hl_gauss_targets(win_prob, self.value_bins)
        if self.mate_classes:
            cls = self.classes(win_prob, mate)
            is_mate = cls >= self.value_bins
            out[is_mate] = 0.0
            out[np.flatnonzero(is_mate), cls[is_mate]] = 1.0
        return out

    def to_dict(self) -> dict[str, int]:
        return {"value_bins": self.value_bins, "mate_classes": self.mate_classes}

    @classmethod
    def from_meta(cls, meta: dict[str, Any], num_classes: int) -> ValueHead:
        spec = meta.get("chess") or {}
        head = cls(spec.get("value_bins", num_classes), spec.get("mate_classes", 0))
        if head.num_classes != num_classes:
            raise ValueError(f"Value head {spec} does not match the model's {num_classes} classes")
        return head

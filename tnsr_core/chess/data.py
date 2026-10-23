"""Chess action-value datasets: Lichess games -> sampled positions -> Stockfish labels.

Every legal move of every sampled position is scored by Stockfish (MultiPV over all legal
moves, a fixed node budget per move) and converted to a win probability for the side to
move. A dataset lives in ``datasets/<name>/``:

    meta.json                provenance (source file + hash, license), Stockfish version and
                             settings, split, counts
    positions.jsonl          the sampled positions (fen, game id, split) - fixed before labeling
    shards/<split>_NNNNN.npz states (N, 77) uint8 char ids, move_offsets (N+1), actions (M)
                             uint16 action indices, win_probs (M) float32, fens (N)

Labeling writes one shard per chunk of positions and skips shards that already exist, so an
interrupted ``tnsr chess prepare`` resumes where it stopped.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import multiprocessing as mp
import os
import random
import shutil
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chess
import chess.engine
import chess.pgn
import numpy as np
import torch

from tnsr_core.paths import datasets_dir, safe_join, validate_name

from .encoding import ChessTokenizer, win_prob_to_bin

CHESS_DATASET_FORMAT = 1
DATASET_KIND = "chess_action_value"
LICHESS_LICENSE = "CC0 1.0 (Lichess open database, https://database.lichess.org)"


class ChessDataError(ValueError):
    pass


def find_stockfish(path: str | None = None) -> str:
    candidates = [path, os.environ.get("TNSR_STOCKFISH"), shutil.which("stockfish"), "/usr/games/stockfish",
                  "/opt/homebrew/bin/stockfish", "/usr/local/bin/stockfish"]  # fmt: skip
    for c in candidates:
        if c and Path(c).is_file() and os.access(c, os.X_OK):
            return c
    raise ChessDataError(
        "Stockfish not found. Install it (macOS: `brew install stockfish`, Debian/Ubuntu: "
        "`apt install stockfish`) or set TNSR_STOCKFISH=/path/to/stockfish."
    )


def cp_to_win_prob(cp: float) -> float:
    """Lichess's centipawn -> expected score model (unrounded)."""
    return 1.0 / (1.0 + math.exp(-0.00368208 * cp))


def score_to_win_prob(score: chess.engine.PovScore, turn: chess.Color) -> float:
    s = score.pov(turn)
    mate = s.mate()
    if mate is not None:
        return 1.0 if mate > 0 else 0.0
    cp = s.score()
    assert cp is not None
    return cp_to_win_prob(cp)


# -- reading games -------------------------------------------------------------------------


def open_pgn(path: str | Path) -> io.TextIOWrapper:
    path = Path(path)
    if not path.is_file():
        raise ChessDataError(f"PGN file not found: {path}")
    raw = path.open("rb")
    if path.suffix == ".zst":
        import zstandard

        stream = zstandard.ZstdDecompressor().stream_reader(raw)
        return io.TextIOWrapper(stream, encoding="utf-8", errors="replace")
    return io.TextIOWrapper(raw, encoding="utf-8", errors="replace")


@dataclass
class SampledPosition:
    fen: str
    game_id: str
    split: str


def position_key(board: chess.Board) -> str:
    """Position identity for de-duplication (ignores move counters)."""
    return board.epd()


def game_split(game_id: str, val_fraction: float, seed: int) -> str:
    h = hashlib.sha256(f"{seed}:{game_id}".encode()).digest()
    return "val" if int.from_bytes(h[:8], "big") / 2**64 < val_fraction else "train"


def sample_positions(
    pgn_path: str | Path,
    max_positions: int,
    *,
    per_game: int = 4,
    val_fraction: float = 0.05,
    seed: int = 0,
    min_elo: int = 0,
    log: Callable[[str], None] = lambda _m: None,
) -> tuple[list[SampledPosition], dict[str, Any]]:
    """Sample up to ``per_game`` distinct positions per game, split by game.

    Validation positions that also occur (as a board position) in the training sample are
    dropped, so the model is never evaluated on positions it was trained on.
    """
    rng = random.Random(seed)
    seen: dict[str, str] = {}  # position key -> split
    out: list[SampledPosition] = []
    games = skipped = 0
    with open_pgn(pgn_path) as fh:
        while len(out) < max_positions:
            game = chess.pgn.read_game(fh)
            if game is None:
                break
            headers = game.headers
            if game.errors or headers.get("Variant", "Standard") != "Standard":
                skipped += 1
                continue
            try:
                elos = [int(headers.get("WhiteElo", "0")), int(headers.get("BlackElo", "0"))]
            except ValueError:
                elos = [0, 0]
            if min(elos) < min_elo:
                skipped += 1
                continue
            moves = list(game.mainline_moves())
            if len(moves) < 2:
                skipped += 1
                continue
            games += 1
            game_id = headers.get("Site") or f"game-{games}"
            split = game_split(game_id, val_fraction, seed)
            plies = sorted(rng.sample(range(len(moves)), min(per_game, len(moves))))
            board = game.board()
            want = set(plies)
            for ply, move in enumerate(moves):
                if ply in want and not board.is_game_over():
                    key = position_key(board)
                    prev = seen.get(key)
                    if prev is None:
                        seen[key] = split
                        out.append(SampledPosition(board.fen(), game_id, split))
                    elif prev == "val" and split == "train":
                        seen[key] = "train"  # train wins; the val copy is removed below
                board.push(move)
            if games % 2000 == 0:
                log(f"read {games:,} games, sampled {len(out):,} positions")
    out = [
        p
        for p in out
        if not (p.split == "val" and seen[position_key(chess.Board(p.fen))] == "train")
    ]
    stats = {"games_used": games, "games_skipped": skipped, "positions": len(out)}
    return out[:max_positions], stats


# -- labeling ------------------------------------------------------------------------------

_ENGINE: chess.engine.SimpleEngine | None = None
_NODES = 2000


def _init_worker(stockfish: str, nodes: int, hash_mb: int) -> None:
    global _ENGINE, _NODES
    _ENGINE = chess.engine.SimpleEngine.popen_uci(stockfish)
    _ENGINE.configure({"Threads": 1, "Hash": hash_mb})
    _NODES = nodes


def label_position(fen: str) -> list[tuple[str, float]] | None:
    """Win probability (for the side to move) after each legal move; None if incomplete."""
    assert _ENGINE is not None, "worker not initialised"
    board = chess.Board(fen)
    legal = [m.uci() for m in board.legal_moves]
    if not legal:
        return None
    infos = _ENGINE.analyse(
        board, chess.engine.Limit(nodes=_NODES * len(legal)), multipv=len(legal)
    )
    values: dict[str, float] = {}
    for info in infos:
        pv = info.get("pv")
        score = info.get("score")
        if pv and score is not None:
            values[pv[0].uci()] = score_to_win_prob(score, board.turn)
    if set(values) != set(legal):
        return None
    return [(u, values[u]) for u in legal]


def _write_shard(
    path: Path, tokenizer: ChessTokenizer, fens: list[str], labels: list[list[tuple[str, float]]]
) -> int:
    states = np.stack([tokenizer.encode_fen(f) for f in fens]).astype(np.uint8)
    offsets = np.zeros(len(fens) + 1, dtype=np.int64)
    actions: list[int] = []
    probs: list[float] = []
    for i, lab in enumerate(labels):
        actions.extend(tokenizer.action_index(u) for u, _ in lab)
        probs.extend(p for _, p in lab)
        offsets[i + 1] = len(actions)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(
        tmp,
        states=states,
        move_offsets=offsets,
        actions=np.asarray(actions, dtype=np.uint16),
        win_probs=np.asarray(probs, dtype=np.float32),
        fens=np.asarray(fens),
    )
    tmp.rename(path)
    return len(actions)


def prepare_chess_dataset(
    name: str,
    pgn_path: str | Path,
    *,
    max_positions: int = 20_000,
    per_game: int = 4,
    val_fraction: float = 0.05,
    nodes: int = 2000,
    workers: int | None = None,
    shard_size: int = 2000,
    seed: int = 0,
    min_elo: int = 0,
    stockfish: str | None = None,
    root: Path | None = None,
    log: Callable[[str], None] = print,
) -> Path:
    """Create (or resume) a labeled dataset. Re-running with the same name resumes labeling."""
    validate_name(name, "dataset name")
    out = safe_join(root or datasets_dir(), name)
    engine_path = find_stockfish(stockfish)
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    pgn_path = Path(pgn_path)
    tokenizer = ChessTokenizer()
    meta_path = out / "meta.json"
    positions_path = out / "positions.jsonl"

    if meta_path.is_file():
        meta = json.loads(meta_path.read_text())
        if meta.get("kind") != DATASET_KIND:
            raise ChessDataError(f"{out} exists and is not a chess dataset")
        if meta["source"]["file"] != pgn_path.name or meta["labeling"]["nodes_per_move"] != nodes:
            raise ChessDataError(f"{out} was prepared from different settings; choose a new name")
        log(f"Resuming {name}")
    else:
        out.mkdir(parents=True, exist_ok=False)
        (out / "shards").mkdir()
        log(f"Sampling up to {max_positions:,} positions from {pgn_path.name}")
        positions, stats = sample_positions(
            pgn_path, max_positions, per_game=per_game, val_fraction=val_fraction, seed=seed,
            min_elo=min_elo, log=log,
        )  # fmt: skip
        if not positions:
            raise ChessDataError("No usable positions found in the PGN file")
        with positions_path.open("w") as fh:
            for p in positions:
                fh.write(json.dumps(p.__dict__) + "\n")
        sha = hashlib.sha256()
        with pgn_path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                sha.update(chunk)
        with chess.engine.SimpleEngine.popen_uci(engine_path) as eng:
            engine_name = eng.id.get("name", "stockfish")
        meta = {
            "format_version": CHESS_DATASET_FORMAT,
            "kind": DATASET_KIND,
            "name": name,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "source": {"file": pgn_path.name, "sha256": sha.hexdigest(), "license": LICHESS_LICENSE},
            "sampling": {"per_game": per_game, "max_positions": max_positions, "seed": seed, "min_elo": min_elo, **stats},
            "split": {"method": "sha256(seed:game Site) per game; val positions seen in train removed", "val_fraction": val_fraction},
            "labeling": {"engine": engine_name, "nodes_per_move": nodes, "value": "win probability for side to move (Lichess cp model; mate = 1/0)"},
            "tokenizer": tokenizer.describe(),
            "shard_size": shard_size,
            "complete": False,
        }  # fmt: skip
        _write_meta(out, meta)

    positions = [
        SampledPosition(**json.loads(line)) for line in positions_path.read_text().splitlines()
    ]
    jobs: list[tuple[Path, list[str]]] = []
    for split in ("train", "val"):
        fens = [p.fen for p in positions if p.split == split]
        for i in range(0, len(fens), shard_size):
            path = out / "shards" / f"{split}_{i // shard_size:05d}.npz"
            if not path.exists():
                jobs.append((path, fens[i : i + shard_size]))
    total = sum(len(f) for _, f in jobs)
    if jobs:
        log(f"Labeling {total:,} positions with {workers} Stockfish workers at {nodes} nodes/move")
        t0 = time.time()
        done = 0
        ctx = mp.get_context("spawn")
        with ctx.Pool(workers, initializer=_init_worker, initargs=(engine_path, nodes, 16)) as pool:
            for path, fens in jobs:
                labels = pool.map(label_position, fens, chunksize=8)
                kept = [(f, lab) for f, lab in zip(fens, labels, strict=True) if lab]
                _write_shard(path, tokenizer, [f for f, _ in kept], [lab for _, lab in kept])
                done += len(fens)
                rate = done / max(1e-9, time.time() - t0)
                log(
                    f"{path.name}: {len(kept)}/{len(fens)} labeled | {done:,}/{total:,} | {rate:.1f} pos/s | ETA {(total - done) / rate / 60:.1f} min"
                )
    meta = json.loads(meta_path.read_text())
    meta["stats"] = _shard_stats(out)
    meta["complete"] = True
    meta["identity"] = f"{name}-{_content_hash(out)}"
    _write_meta(out, meta)
    log(f"Dataset ready: {out}")
    return out


def _write_meta(out: Path, meta: dict[str, Any]) -> None:
    tmp = out / ".meta.json.tmp"
    tmp.write_text(json.dumps(meta, indent=2))
    os.replace(tmp, out / "meta.json")


def _shard_files(out: Path, split: str) -> list[Path]:
    return sorted((out / "shards").glob(f"{split}_*.npz"))


def _shard_stats(out: Path) -> dict[str, Any]:
    stats: dict[str, Any] = {}
    for split in ("train", "val"):
        n_pos = n_rec = 0
        for f in _shard_files(out, split):
            with np.load(f) as z:
                n_pos += len(z["states"])
                n_rec += len(z["actions"])
        stats[f"{split}_positions"] = n_pos
        stats[f"{split}_records"] = n_rec
    return stats


def _content_hash(out: Path) -> str:
    sha = hashlib.sha256()
    for f in sorted((out / "shards").glob("*.npz")):
        sha.update(f.name.encode())
        sha.update(f.read_bytes())
    return sha.hexdigest()[:12]


# -- loading for training ------------------------------------------------------------------


def resolve_chess_dataset(name_or_path: str) -> Path:
    path = Path(name_or_path)
    if not (path.is_dir() and (path / "meta.json").is_file()):
        validate_name(name_or_path, "dataset name")
        path = safe_join(datasets_dir(), name_or_path)
    meta_path = path / "meta.json"
    if not meta_path.is_file():
        raise ChessDataError(
            f"Chess dataset {name_or_path!r} not found (create it with: tnsr chess prepare)"
        )
    meta = json.loads(meta_path.read_text())
    if meta.get("kind") != DATASET_KIND:
        raise ChessDataError(f"{path} is not a chess dataset")
    if not meta.get("complete"):
        raise ChessDataError(
            f"{path} is not fully labeled yet; re-run `tnsr chess prepare` to resume"
        )
    return path


class ChessSplit:
    """All (position, move, win probability) records of one split, held in memory."""

    def __init__(self, path: Path, split: str, tokenizer: ChessTokenizer):
        states, state_idx, actions, probs, fens = [], [], [], [], []
        base = 0
        for f in _shard_files(path, split):
            with np.load(f) as z:
                s = z["states"]
                offs = z["move_offsets"]
                states.append(s)
                counts = np.diff(offs)
                state_idx.append(np.repeat(np.arange(base, base + len(s)), counts))
                actions.append(z["actions"])
                probs.append(z["win_probs"])
                fens.extend(z["fens"].tolist())
                base += len(s)
        if not states:
            raise ChessDataError(f"No {split} shards in {path}")
        self.tokenizer = tokenizer
        self.states = torch.from_numpy(np.concatenate(states).astype(np.int64))
        self.state_idx = torch.from_numpy(np.concatenate(state_idx).astype(np.int64))
        self.actions = torch.from_numpy(np.concatenate(actions).astype(np.int64))
        self.win_probs = torch.from_numpy(np.concatenate(probs).astype(np.float32))
        self.fens = fens

    @property
    def num_records(self) -> int:
        return int(self.actions.numel())

    @property
    def num_positions(self) -> int:
        return int(self.states.size(0))

    def batch(
        self, idx: torch.Tensor, num_bins: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Token ids (B, 78), target bins (B,), target win probabilities (B,)."""
        st = self.states[self.state_idx[idx]]
        act = (self.actions[idx] + self.tokenizer.action_offset).unsqueeze(1)
        x = torch.cat([st, act], dim=1)
        wp = self.win_probs[idx]
        y = torch.from_numpy(win_prob_to_bin(wp.numpy(), num_bins))
        return x, y, wp


def iter_positions(split: ChessSplit) -> Iterator[tuple[str, list[tuple[str, float]]]]:
    """(fen, [(uci, win_prob), ...]) per position, for move-quality evaluation."""
    order = torch.argsort(split.state_idx, stable=True)
    sidx = split.state_idx[order]
    bounds = torch.searchsorted(sidx, torch.arange(split.num_positions + 1))
    for i in range(split.num_positions):
        rec = order[bounds[i] : bounds[i + 1]]
        moves = [
            (split.tokenizer.action_from_index(int(split.actions[r])), float(split.win_probs[r]))
            for r in rec
        ]
        yield split.fens[i], moves


def build_dataset_from_labels(
    name: str,
    items: list[tuple[str, list[tuple[str, float]], str]],
    *,
    source: str = "custom",
    labeling: str = "custom",
    root: Path | None = None,
    shard_size: int = 2000,
) -> Path:
    """Write a complete chess dataset from (fen, [(uci, win_prob)], split) items.

    Used by tests and for importing positions labeled elsewhere (e.g. on a GPU box).
    """
    validate_name(name, "dataset name")
    out = safe_join(root or datasets_dir(), name)
    out.mkdir(parents=True, exist_ok=False)
    (out / "shards").mkdir()
    tokenizer = ChessTokenizer()
    for split in ("train", "val"):
        rows = [(f, lab) for f, lab, s in items if s == split]
        for i in range(0, len(rows), shard_size):
            chunk = rows[i : i + shard_size]
            _write_shard(out / "shards" / f"{split}_{i // shard_size:05d}.npz", tokenizer,
                         [f for f, _ in chunk], [lab for _, lab in chunk])  # fmt: skip
    meta = {
        "format_version": CHESS_DATASET_FORMAT,
        "kind": DATASET_KIND,
        "name": name,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source": {"file": source, "license": "see source"},
        "labeling": {"engine": labeling},
        "tokenizer": tokenizer.describe(),
        "complete": True,
        "stats": {},
    }
    _write_meta(out, meta)
    meta["stats"] = _shard_stats(out)
    meta["identity"] = f"{name}-{_content_hash(out)}"
    _write_meta(out, meta)
    return out

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

from .encoding import ChessTokenizer, ValueHead

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
    out = out[:max_positions]
    if not any(p.split == "val" for p in out):
        # Tiny corpora can hash every game into train: move the last game wholesale to val.
        game_ids = list(dict.fromkeys(p.game_id for p in out))
        if len(game_ids) < 2:
            raise ChessDataError("Need positions from at least two games to make a train/val split")
        moved = game_ids[-1]
        out = [SampledPosition(p.fen, p.game_id, "val") if p.game_id == moved else p for p in out]
        log(f"No game hashed into validation; using game {moved} as the validation split")
    if not any(p.split == "train" for p in out):
        raise ChessDataError("All sampled positions fell into validation; lower val_fraction")
    stats = {"games_used": games, "games_skipped": skipped, "positions": len(out)}
    return out, stats


# -- labeling ------------------------------------------------------------------------------

_ENGINE: chess.engine.SimpleEngine | None = None
_NODES = 2000


def _init_worker(stockfish: str, nodes: int, hash_mb: int) -> None:
    global _ENGINE, _NODES
    _ENGINE = chess.engine.SimpleEngine.popen_uci(stockfish)
    _ENGINE.configure({"Threads": 1, "Hash": hash_mb})
    _NODES = nodes


MATE_UNKNOWN = 99  # mate whose distance was not recorded (datasets from before mate tracking)


def score_to_mate(score: chess.engine.PovScore, turn: chess.Color) -> int:
    """+n: the side to move mates in n (counting this move); -n: it gets mated in n; 0: no mate."""
    mate = score.pov(turn).mate()
    return 0 if mate is None else int(mate)


Label = tuple[str, float, int]  # (uci, win probability for the mover, mate distance)


def label_position(fen: str) -> list[Label] | None:
    """Win probability and mate distance (for the side to move) after each legal move."""
    assert _ENGINE is not None, "worker not initialised"
    board = chess.Board(fen)
    legal = [m.uci() for m in board.legal_moves]
    if not legal:
        return None
    infos = _ENGINE.analyse(
        board, chess.engine.Limit(nodes=_NODES * len(legal)), multipv=len(legal)
    )
    values: dict[str, tuple[float, int]] = {}
    for info in infos:
        pv = info.get("pv")
        score = info.get("score")
        if pv and score is not None:
            values[pv[0].uci()] = (
                score_to_win_prob(score, board.turn),
                score_to_mate(score, board.turn),
            )
    if set(values) != set(legal):
        return None
    return [(u, *values[u]) for u in legal]


def _write_shard(
    path: Path, tokenizer: ChessTokenizer, fens: list[str], labels: list[list[Any]]
) -> int:
    """``labels`` rows are (uci, win_prob) or (uci, win_prob, mate)."""
    states = np.stack([tokenizer.encode_fen(f) for f in fens]).astype(np.uint8)
    offsets = np.zeros(len(fens) + 1, dtype=np.int64)
    actions: list[int] = []
    probs: list[float] = []
    mates: list[int] = []
    for i, lab in enumerate(labels):
        for row in lab:
            actions.append(tokenizer.action_index(row[0]))
            probs.append(row[1])
            mates.append(row[2] if len(row) > 2 else _infer_mate(row[1]))
        offsets[i + 1] = len(actions)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(
        tmp,
        states=states,
        move_offsets=offsets,
        actions=np.asarray(actions, dtype=np.uint16),
        win_probs=np.asarray(probs, dtype=np.float32),
        mates=np.asarray(mates, dtype=np.int16),
        fens=np.asarray(fens),
    )
    tmp.rename(path)
    return len(actions)


def _infer_mate(win_prob: float) -> int:
    return MATE_UNKNOWN if win_prob >= 1.0 else -MATE_UNKNOWN if win_prob <= 0.0 else 0


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
    endgames: int = 0,
    stockfish: str | None = None,
    source_license: str = LICHESS_LICENSE,
    root: Path | None = None,
    log: Callable[[str], None] = print,
) -> Path:
    """Create (or resume) a labeled dataset. Re-running with the same name resumes labeling.

    ``endgames`` adds that many generated endgame positions (see ``endgames.py``), split into
    train/val by hashing the position.
    """
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
        if endgames:
            from .endgames import generate_endgames

            known = {position_key(chess.Board(p.fen)) for p in positions}
            added = 0
            for fen in generate_endgames(endgames, seed):
                if position_key(chess.Board(fen)) in known:
                    continue
                split = game_split(fen, val_fraction, seed)
                positions.append(SampledPosition(fen, "endgame", split))
                added += 1
            stats["endgame_positions"] = added
            log(f"Added {added:,} generated endgame positions")
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
            "source": {"file": pgn_path.name, "sha256": sha.hexdigest(), "license": source_license},
            "sampling": {"per_game": per_game, "max_positions": max_positions, "seed": seed, "min_elo": min_elo, "endgames": endgames, **stats},
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

    def __init__(
        self,
        path: Path | list[Path],
        split: str,
        tokenizer: ChessTokenizer,
        load_fens: bool = True,
    ):
        """``load_fens=False`` skips the FEN strings (only evaluation needs them), which is
        most of the memory for large training sets."""
        paths = path if isinstance(path, list) else [path]
        states, state_idx, actions, probs, mates, fens = [], [], [], [], [], []
        base = 0
        for f in [f for p in paths for f in _shard_files(p, split)]:
            with np.load(f) as z:
                s = z["states"]
                offs = z["move_offsets"]
                states.append(s)
                counts = np.diff(offs)
                state_idx.append(np.repeat(np.arange(base, base + len(s)), counts))
                actions.append(z["actions"])
                probs.append(z["win_probs"])
                if "mates" in z:
                    mates.append(z["mates"].astype(np.int64))
                else:  # shards from before mate tracking: only "mate, distance unknown"
                    wp = z["win_probs"]
                    mates.append(
                        np.where(wp >= 1.0, MATE_UNKNOWN, np.where(wp <= 0.0, -MATE_UNKNOWN, 0))
                    )
                if load_fens:
                    fens.extend(z["fens"].tolist())
                base += len(s)
        if not states:
            raise ChessDataError(f"No {split} shards in {', '.join(map(str, paths))}")
        self.tokenizer = tokenizer
        # uint8 in memory (77 bytes per position); widened to int64 per batch.
        self.states = torch.from_numpy(np.concatenate(states).astype(np.uint8))
        self.state_idx = torch.from_numpy(np.concatenate(state_idx).astype(np.int64))
        self.actions = torch.from_numpy(np.concatenate(actions).astype(np.int64))
        self.win_probs = torch.from_numpy(np.concatenate(probs).astype(np.float32))
        self.mates = torch.from_numpy(np.concatenate(mates).astype(np.int64))
        self.fens: list[str] | None = fens if load_fens else None
        # Records of position i are rows pos_offsets[i] : pos_offsets[i + 1] (shards store them
        # contiguously), which lets a batch hold whole positions.
        counts = torch.bincount(self.state_idx, minlength=self.num_positions)
        self.pos_offsets = torch.zeros(self.num_positions + 1, dtype=torch.int64)
        self.pos_offsets[1:] = torch.cumsum(counts, 0)

    @property
    def num_records(self) -> int:
        return int(self.actions.numel())

    @property
    def num_positions(self) -> int:
        return int(self.states.size(0))

    def batch(
        self, idx: torch.Tensor, head: ValueHead | int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Token ids (B, 78), target classes (B,), target win probabilities (B,)."""
        if isinstance(head, int):
            head = ValueHead(head)
        st = self.states[self.state_idx[idx]].long()
        if self.tokenizer.move_marks:
            st = self.tokenizer.mark_moves(st, self.actions[idx])
        act = (self.actions[idx] + self.tokenizer.action_offset).unsqueeze(1)
        x = torch.cat([st, act], dim=1)
        wp = self.win_probs[idx]
        y = torch.from_numpy(head.classes(wp.numpy(), self.mates[idx].numpy()))
        return x, y, wp

    def position_batch(
        self, positions: torch.Tensor, head: ValueHead, soft: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Whole positions for shared-board models: token ids (P, 77 + K) holding each position
        and its K (padded) labeled moves, and targets: classes (P, K) with -100 for padding,
        or with ``soft`` HL-Gauss rows (P, K, C) that are all zero for padding."""
        start = self.pos_offsets[positions]
        count = self.pos_offsets[positions + 1] - start
        k = int(count.max())
        slot = torch.arange(k).unsqueeze(0)
        valid = slot < count.unsqueeze(1)
        rec = torch.where(valid, start.unsqueeze(1) + slot, start.unsqueeze(1))
        acts = torch.where(valid, self.actions[rec] + self.tokenizer.action_offset, 0)
        x = torch.cat([self.states[positions].long(), acts], dim=1)
        flat = rec.reshape(-1)
        if soft:
            t = self.soft_targets(flat, head).reshape(len(positions), k, -1)
            return x, t * valid.unsqueeze(-1)
        y = torch.from_numpy(
            head.classes(self.win_probs[flat].numpy(), self.mates[flat].numpy())
        ).reshape(len(positions), k)
        return x, torch.where(valid, y, -100)

    def soft_targets(self, idx: torch.Tensor, head: ValueHead) -> torch.Tensor:
        return torch.from_numpy(
            head.soft_targets(self.win_probs[idx].numpy(), self.mates[idx].numpy())
        ).float()


def iter_positions(split: ChessSplit) -> Iterator[tuple[str, list[tuple[str, float]]]]:
    """(fen, [(uci, win_prob), ...]) per position, for move-quality evaluation."""
    fens = split.fens
    if fens is None:
        raise ChessDataError("This split was loaded without FENs (load_fens=False)")
    order = torch.argsort(split.state_idx, stable=True)
    sidx = split.state_idx[order]
    bounds = torch.searchsorted(sidx, torch.arange(split.num_positions + 1))
    for i in range(split.num_positions):
        rec = order[bounds[i] : bounds[i + 1]]
        moves = [
            (split.tokenizer.action_from_index(int(split.actions[r])), float(split.win_probs[r]))
            for r in rec
        ]
        yield fens[i], moves


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

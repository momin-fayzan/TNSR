"""Import DeepMind's ChessBench action-value data (Ruoss et al., 2024) as TNSR datasets.

ChessBench labels every legal move of positions from ~10M Lichess games with Stockfish 16
(50 ms per move) win probabilities: 2,148 training files of ~1.3 GB, ~15B records in all.
The data is CC-BY 4.0 (with CC0 Lichess portions); credit: Google DeepMind,
https://github.com/google-deepmind/searchless_chess.

File format (``.bag``): records are concatenated, followed by an index of little-endian uint64
end offsets; the last 8 bytes give the index start. An action-value record is an Apache Beam
``TupleCoder(StrUtf8, StrUtf8, Float)``: varint length + FEN, varint length + UCI move,
big-endian float64 win probability.
"""

from __future__ import annotations

import hashlib
import json
import mmap
import struct
import time
import urllib.request
from collections.abc import Callable, Iterator
from pathlib import Path

from tnsr_core.paths import datasets_dir, safe_join, validate_name

from .data import (
    CHESS_DATASET_FORMAT,
    DATASET_KIND,
    ChessDataError,
    _content_hash,
    _shard_stats,
    _write_meta,
    _write_shard,
)
from .encoding import ChessTokenizer

BASE_URL = "https://storage.googleapis.com/searchless_chess/data"
NUM_TRAIN_FILES = 2148
LICENSE = (
    "CC-BY 4.0 (portions CC0 from lichess.org); ChessBench by Google DeepMind, "
    "https://github.com/google-deepmind/searchless_chess"
)


def train_file_url(index: int) -> str:
    if not 0 <= index < NUM_TRAIN_FILES:
        raise ValueError(f"ChessBench train file index must be in 0..{NUM_TRAIN_FILES - 1}")
    return f"{BASE_URL}/train/action_value-{index:05d}-of-{NUM_TRAIN_FILES:05d}_data.bag"


def download_train_file(index: int, log: Callable[[str], None] = print) -> Path:
    """Download one training file to datasets/raw/chessbench/ (skipped if already there)."""
    out_dir = datasets_dir() / "raw" / "chessbench"
    out_dir.mkdir(parents=True, exist_ok=True)
    url = train_file_url(index)
    out = out_dir / url.rsplit("/", 1)[1]
    if out.exists():
        return out
    tmp = out.with_suffix(".partial")
    log(f"Downloading {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "tnsr"})
    with urllib.request.urlopen(req, timeout=60) as resp, tmp.open("wb") as fh:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        last = time.time()
        while chunk := resp.read(1 << 22):
            fh.write(chunk)
            done += len(chunk)
            if time.time() - last > 15:
                log(f"  {done / 1e6:,.0f}/{total / 1e6:,.0f} MB")
                last = time.time()
    if total and tmp.stat().st_size != total:
        raise ChessDataError(f"Incomplete download of {url}; re-run to retry")
    tmp.rename(out)
    return out


def _varint(buf: bytes | mmap.mmap, pos: int) -> tuple[int, int]:
    value = shift = 0
    while True:
        b = buf[pos]
        pos += 1
        value |= (b & 0x7F) << shift
        if not b & 0x80:
            return value, pos
        shift += 7


def decode_record(rec: bytes) -> tuple[str, str, float]:
    n, pos = _varint(rec, 0)
    fen = rec[pos : pos + n].decode()
    pos += n
    n, pos = _varint(rec, pos)
    move = rec[pos : pos + n].decode()
    pos += n
    if len(rec) - pos != 8:
        raise ChessDataError("Unexpected ChessBench record layout (not an action-value file?)")
    (win_prob,) = struct.unpack(">d", rec[pos:])
    return fen, move, win_prob


def read_bag(path: Path) -> Iterator[tuple[str, str, float]]:
    """(fen, uci, win_prob) records of a ChessBench action-value ``.bag`` file."""
    with path.open("rb") as fh, mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as m:
        if len(m) < 8:
            raise ChessDataError(f"{path} is not a bag file")
        index_start = int.from_bytes(m[-8:], "little")
        if not 0 < index_start <= len(m) - 8 or (len(m) - index_start) % 8:
            raise ChessDataError(f"{path} is not a bag file")
        n = (len(m) - index_start) // 8
        start = 0
        for i in range(n):
            end = int.from_bytes(m[index_start + 8 * i : index_start + 8 * i + 8], "little")
            yield decode_record(m[start:end])
            start = end


def _key(fen: str) -> str:
    return " ".join(fen.split()[:4])


def import_chessbench(
    name: str,
    files: list[Path],
    *,
    max_records: int | None = None,
    exclude_fens: set[str] | None = None,
    val_fraction: float = 0.002,
    chunk_records: int = 2_000_000,
    shard_positions: int = 20_000,
    root: Path | None = None,
    log: Callable[[str], None] = print,
) -> Path:
    """Write ChessBench records as a TNSR chess dataset.

    Records are grouped by position within chunks of ``chunk_records`` (files are shuffled, so
    a position can appear in more than one chunk; each copy is a separate training state).
    Positions in ``exclude_fens`` (compared on placement, side, castling and en passant), for
    example evaluation positions, are dropped. ChessBench has no mate distances, so mates are
    stored as plain win probability 1 / 0.
    """
    validate_name(name, "dataset name")
    out = safe_join(root or datasets_dir(), name)
    out.mkdir(parents=True, exist_ok=False)
    (out / "shards").mkdir()
    tokenizer = ChessTokenizer()
    excluded = {_key(f) for f in exclude_fens or set()}
    known_moves = set(tokenizer.actions)
    counters = {"train": 0, "val": 0}
    stats = {"records_read": 0, "records_excluded": 0, "records_unknown_move": 0}
    t0 = time.time()

    def flush(groups: dict[str, list[tuple[str, float, int]]]) -> None:
        by_split: dict[str, list[tuple[str, list[tuple[str, float, int]]]]] = {
            "train": [],
            "val": [],
        }
        for fen, rows in groups.items():
            h = hashlib.sha256(_key(fen).encode()).digest()
            split = "val" if int.from_bytes(h[:8], "big") / 2**64 < val_fraction else "train"
            by_split[split].append((fen, rows))
        for split, items in by_split.items():
            for i in range(0, len(items), shard_positions):
                part = items[i : i + shard_positions]
                path = out / "shards" / f"{split}_{counters[split]:05d}.npz"
                _write_shard(path, tokenizer, [f for f, _ in part], [r for _, r in part])
                counters[split] += 1

    groups: dict[str, list[tuple[str, float, int]]] = {}
    in_chunk = 0
    done = False
    for path in files:
        for fen, move, wp in read_bag(path):
            stats["records_read"] += 1
            if excluded and _key(fen) in excluded:
                stats["records_excluded"] += 1
                continue
            if move not in known_moves:
                stats["records_unknown_move"] += 1
                continue
            groups.setdefault(fen, []).append((move, wp, 0))
            in_chunk += 1
            if in_chunk >= chunk_records:
                flush(groups)
                groups, in_chunk = {}, 0
                rate = stats["records_read"] / max(1e-9, time.time() - t0)
                log(f"{stats['records_read']:,} records read ({rate:,.0f}/s)")
            if max_records and stats["records_read"] >= max_records:
                done = True
                break
        if done:
            break
    if groups:
        flush(groups)
    if not counters["train"]:
        raise ChessDataError("No ChessBench records were imported")
    if not counters["val"]:
        raise ChessDataError("No position hashed into validation; raise val_fraction")
    meta = {
        "format_version": CHESS_DATASET_FORMAT,
        "kind": DATASET_KIND,
        "name": name,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source": {"file": ", ".join(p.name for p in files), "license": LICENSE},
        "sampling": {"max_records": max_records, "excluded_positions": len(excluded), **stats},
        "split": {"method": "sha256(position) per position", "val_fraction": val_fraction},
        "labeling": {
            "engine": "Stockfish 16 (ChessBench, 50 ms per move)",
            "value": "win probability for side to move; no mate distances",
        },
        "tokenizer": tokenizer.describe(),
        "complete": True,
    }
    _write_meta(out, meta)
    meta["stats"] = _shard_stats(out)
    meta["identity"] = f"{name}-{_content_hash(out)}"
    _write_meta(out, meta)
    log(f"Dataset ready: {out} {json.dumps(meta['stats'])}")
    return out

"""Measuring playing strength: held-out move quality, Lichess puzzles, games vs Stockfish."""

from __future__ import annotations

import csv
import io
import json
import random
import time
import urllib.request
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

import chess
import chess.engine

from tnsr_core.paths import datasets_dir

from .data import LICHESS_LICENSE, find_stockfish
from .engine import ChessEngine

PUZZLE_URL = "https://database.lichess.org/lichess_db_puzzle.csv.zst"
PUZZLE_DIR = "chess-puzzles"
RATING_BUCKETS = [(0, 1000), (1000, 1500), (1500, 2000), (2000, 2500), (2500, 4000)]

# Short, common opening lines (UCI). Each is played twice with colours swapped so that a
# deterministic engine still produces varied games.
OPENINGS = [
    "e2e4 e7e5 g1f3 b8c6", "e2e4 c7c5 g1f3 d7d6", "d2d4 d7d5 c2c4 e7e6", "d2d4 g8f6 c2c4 g7g6",
    "e2e4 e7e6 d2d4 d7d5", "e2e4 c7c6 d2d4 d7d5", "c2c4 e7e5 b1c3 g8f6", "g1f3 d7d5 g2g3 g8f6",
    "e2e4 e7e5 f1c4 g8f6", "d2d4 g8f6 c2c4 e7e6", "e2e4 d7d5 e4d5 d8d5", "d2d4 d7d5 c1f4 g8f6",
]  # fmt: skip


# -- puzzles -----------------------------------------------------------------------------------


def puzzles_path() -> Path:
    return datasets_dir() / PUZZLE_DIR / "puzzles.csv"


def download_puzzles(
    count: int = 2000, url: str = PUZZLE_URL, log: Callable[[str], None] = print
) -> Path:
    """Stream the Lichess puzzle database and keep the first ``count`` puzzles.

    Only the needed prefix of the ~300 MB compressed file is downloaded.
    """
    import zstandard

    out = puzzles_path()
    if out.is_file():
        raise FileExistsError(f"{out} already exists; delete it to download again")
    out.parent.mkdir(parents=True, exist_ok=True)
    log(f"Streaming {count:,} puzzles from {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "tnsr"})
    rows: list[list[str]] = []
    with urllib.request.urlopen(req, timeout=60) as resp:
        reader = io.TextIOWrapper(
            zstandard.ZstdDecompressor().stream_reader(resp), encoding="utf-8"
        )
        for row in csv.reader(reader):
            rows.append(row)
            if len(rows) > count:  # header + count
                break
    tmp = out.with_suffix(".tmp")
    with tmp.open("w", newline="") as fh:
        csv.writer(fh).writerows(rows)
    tmp.rename(out)
    (out.parent / "meta.json").write_text(json.dumps({
        "source": url, "license": LICHESS_LICENSE, "count": len(rows) - 1,
        "downloaded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "note": "first rows of the file; puzzle order in the file is by puzzle id, not rating",
    }, indent=2))  # fmt: skip
    log(f"Saved {len(rows) - 1:,} puzzles to {out}")
    return out


def load_puzzles(limit: int | None = None) -> list[dict[str, Any]]:
    path = puzzles_path()
    if not path.is_file():
        raise FileNotFoundError("No puzzles downloaded yet. Run: tnsr chess puzzles --count 2000")
    with path.open() as fh:
        rows = list(csv.DictReader(fh))
    return rows[:limit] if limit else rows


def _gives_mate(board: chess.Board, move: chess.Move) -> bool:
    board.push(move)
    mate = board.is_checkmate()
    board.pop()
    return mate


def solve_puzzle(engine: ChessEngine, fen: str, moves: list[str]) -> bool:
    """Lichess format: moves[0] is the opponent's setup move; the engine must find every
    following move of its own side. Any mating move counts where the solution mates."""
    board = chess.Board(fen)
    board.push_uci(moves[0])
    for i in range(1, len(moves), 2):
        expected = chess.Move.from_uci(moves[i])
        played = engine.best_move(board)
        if played != expected and not (_gives_mate(board, expected) and _gives_mate(board, played)):
            return False
        board.push(expected)
        if i + 1 < len(moves):
            board.push_uci(moves[i + 1])
    return True


def evaluate_puzzles(engine: ChessEngine, limit: int | None = 1000) -> dict[str, Any]:
    puzzles = load_puzzles(limit)
    solved: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    total = 0
    t0 = time.perf_counter()
    for p in puzzles:
        rating = int(p["Rating"])
        bucket = next(f"{lo}-{hi}" for lo, hi in RATING_BUCKETS if lo <= rating < hi)
        ok = solve_puzzle(engine, p["FEN"], p["Moves"].split())
        solved[bucket][0] += int(ok)
        solved[bucket][1] += 1
        total += int(ok)
    return {
        "accuracy": round(total / max(1, len(puzzles)), 4),
        "solved": total,
        "total": len(puzzles),
        "by_rating": {
            k: {"solved": v[0], "total": v[1], "accuracy": round(v[0] / v[1], 4)}
            for k, v in sorted(solved.items())
        },
        "seconds": round(time.perf_counter() - t0, 1),
    }


# -- games vs Stockfish ------------------------------------------------------------------------


def expected_score(rating: float, opponent: float) -> float:
    return 1.0 / (1.0 + 10 ** ((opponent - rating) / 400))


def performance_rating(results: list[tuple[float, float]]) -> float | None:
    """Rating R that makes the expected total score equal the actual one (per-game opponents).

    ``results`` holds (opponent_rating, score). None when the score is 0% or 100% (unbounded).
    """
    if not results:
        return None
    total = sum(s for _, s in results)
    if total <= 0 or total >= len(results):
        return None
    lo, hi = -1000.0, 5000.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if sum(expected_score(mid, o) for o, _ in results) < total:
            lo = mid
        else:
            hi = mid
    return round((lo + hi) / 2, 1)


def play_game(
    engine: ChessEngine,
    opponent: chess.engine.SimpleEngine,
    opening: str,
    tnsr_white: bool,
    movetime: float,
    max_plies: int = 400,
) -> tuple[float, str]:
    """Return (TNSR score 1/0.5/0, termination)."""
    board = chess.Board()
    for uci in opening.split():
        board.push_uci(uci)
    while not board.is_game_over(claim_draw=True) and board.ply() < max_plies:
        if (board.turn == chess.WHITE) == tnsr_white:
            board.push(engine.best_move(board))
        else:
            reply = opponent.play(board, chess.engine.Limit(time=movetime)).move
            assert reply is not None
            board.push(reply)
    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        return 0.5, "max_plies"
    if outcome.winner is None:
        return 0.5, outcome.termination.name.lower()
    return (
        1.0 if (outcome.winner == chess.WHITE) == tnsr_white else 0.0
    ), outcome.termination.name.lower()


def play_match(
    engine: ChessEngine,
    levels: list[int],
    games_per_level: int = 12,
    movetime: float = 0.05,
    stockfish: str | None = None,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Games against Stockfish limited with UCI_LimitStrength/UCI_Elo at each level."""
    path = find_stockfish(stockfish)
    rows = []
    all_results: list[tuple[float, float]] = []
    with chess.engine.SimpleEngine.popen_uci(path) as sf:
        opts = sf.options
        elo_opt = opts.get("UCI_Elo")
        min_elo = int(elo_opt.min) if elo_opt and elo_opt.min is not None else 1320
        max_elo = int(elo_opt.max) if elo_opt and elo_opt.max is not None else 3190
        engine_name = sf.id.get("name", "stockfish")
        for level in levels:
            level = max(min_elo, min(max_elo, level))
            sf.configure({"UCI_LimitStrength": True, "UCI_Elo": level, "Threads": 1, "Hash": 16})
            w = d = l_ = 0
            for g in range(games_per_level):
                opening = OPENINGS[(g // 2) % len(OPENINGS)]
                score, _term = play_game(
                    engine, sf, opening, tnsr_white=(g % 2 == 0), movetime=movetime
                )
                w, d, l_ = w + (score == 1.0), d + (score == 0.5), l_ + (score == 0.0)
                all_results.append((float(level), score))
            score_pct = (w + 0.5 * d) / games_per_level
            rows.append(
                {
                    "stockfish_elo": level,
                    "wins": w,
                    "draws": d,
                    "losses": l_,
                    "score": round(score_pct, 3),
                }
            )
            log(f"vs Stockfish UCI_Elo {level}: +{w} ={d} -{l_} ({score_pct:.0%})")
    perf = performance_rating(all_results)
    return {
        "opponent": engine_name,
        "movetime_seconds": movetime,
        "games": len(all_results),
        "levels": rows,
        "elo_estimate": perf,
        "elo_note": (
            "Performance rating against Stockfish's UCI_Elo scale (calibrated by the Stockfish team "
            "against CCRL-style ratings, not Lichess ratings). None means the result was 0% or 100%, "
            "so the rating is only bounded from one side."
        ),
    }


def random_mover_baseline(engine: ChessEngine, games: int = 10, seed: int = 0) -> dict[str, Any]:
    """Sanity check: the engine should beat a uniformly random mover almost every time."""
    rng = random.Random(seed)
    w = d = lost = 0
    for g in range(games):
        board = chess.Board()
        tnsr_white = g % 2 == 0
        while not board.is_game_over(claim_draw=True) and board.ply() < 400:
            if (board.turn == chess.WHITE) == tnsr_white:
                board.push(engine.best_move(board))
            else:
                board.push(rng.choice(list(board.legal_moves)))
        outcome = board.outcome(claim_draw=True)
        if outcome is None or outcome.winner is None:
            d += 1
        elif (outcome.winner == chess.WHITE) == tnsr_white:
            w += 1
        else:
            lost += 1
    return {"games": games, "wins": w, "draws": d, "losses": lost}


def evaluation_summary(
    puzzles: dict | None, match: dict | None, move_quality: dict | None
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    if puzzles:
        metrics["puzzle_accuracy"] = puzzles["accuracy"]
    if match:
        metrics["elo_estimate"] = match["elo_estimate"]
    if move_quality:
        metrics.update(move_quality)
    return metrics

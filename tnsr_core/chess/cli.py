"""``tnsr chess`` commands."""

from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

chess_app = typer.Typer(
    help="Chess engine: data, training metrics, evaluation, play.", no_args_is_help=True
)
console = Console()
err = Console(stderr=True)

LICHESS_MONTH_URL = (
    "https://database.lichess.org/standard/lichess_db_standard_rated_{month}.pgn.zst"
)


def _fail(message: str) -> None:
    err.print(f"[bold red]error:[/] {message}")
    raise typer.Exit(1)


def _engine(model: str | None, device: str):
    from tnsr_core.checkpoints import (
        CheckpointError,
        discover_checkpoints,
        preferred_checkpoint,
        resolve_checkpoint,
    )
    from tnsr_core.paths import tnsr_home

    from .engine import ChessEngine

    try:
        if model:
            path = resolve_checkpoint(model)
        else:
            chess_models = [
                c
                for c in discover_checkpoints()
                if (c.get("tokenizer") or {}).get("kind") == "chess"
            ]
            choice = preferred_checkpoint(chess_models)
            if choice is None:
                _fail("No chess checkpoints found. Train one: tnsr train --config chess-mini")
                raise AssertionError
            path = tnsr_home() / choice["id"]
        return ChessEngine.from_checkpoint(path, device)
    except (CheckpointError, ValueError) as exc:
        _fail(str(exc))
        raise


@chess_app.command("download")
def download(
    month: Annotated[
        str, typer.Option(help="Lichess database month, e.g. 2013-01 (older months are smaller)")
    ] = "2013-01",
) -> None:
    """Download one month of rated standard games from the Lichess open database (CC0)."""
    from tnsr_core.paths import datasets_dir

    out_dir = datasets_dir() / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    url = LICHESS_MONTH_URL.format(month=month)
    out = out_dir / url.rsplit("/", 1)[1]
    if out.exists():
        console.print(f"Already downloaded: {out}")
        return
    tmp = out.with_suffix(".partial")
    console.print(f"Downloading {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "tnsr"})
    with urllib.request.urlopen(req, timeout=60) as resp, tmp.open("wb") as fh:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        while chunk := resp.read(1 << 20):
            fh.write(chunk)
            done += len(chunk)
            if total:
                err.print(f"\r{done / 1e6:,.0f}/{total / 1e6:,.0f} MB", end="")
    err.print()
    tmp.rename(out)
    console.print(f"Saved {out}")


@chess_app.command("prepare")
def prepare(
    pgn: Annotated[
        Path | None, typer.Argument(help="PGN or .pgn.zst file (or use --month)")
    ] = None,
    month: Annotated[
        str | None,
        typer.Option(help="Download (if needed) and use this Lichess month, e.g. 2013-01"),
    ] = None,
    name: Annotated[str, typer.Option(help="Dataset name under datasets/")] = "chess-demo",
    max_positions: Annotated[int, typer.Option(help="Positions to sample and label")] = 20_000,
    per_game: Annotated[int, typer.Option(help="Positions sampled per game")] = 2,
    nodes: Annotated[
        int, typer.Option(help="Stockfish nodes per legal move (label quality)")
    ] = 2000,
    workers: Annotated[int, typer.Option(help="Parallel Stockfish processes (0 = CPUs - 1)")] = 0,
    val_fraction: Annotated[float, typer.Option()] = 0.05,
    min_elo: Annotated[
        int, typer.Option(help="Skip games where either player is below this rating")
    ] = 0,
    endgames: Annotated[
        int,
        typer.Option(help="Add this many generated endgame positions (teaches converting wins)"),
    ] = 0,
    seed: Annotated[int, typer.Option()] = 0,
) -> None:
    """Sample positions and label every legal move with Stockfish (resumable)."""
    from .data import ChessDataError, prepare_chess_dataset

    if pgn is None:
        if month is None:
            _fail("Give a PGN file or --month")
            return
        download(month)
        from tnsr_core.paths import datasets_dir

        pgn = datasets_dir() / "raw" / LICHESS_MONTH_URL.format(month=month).rsplit("/", 1)[1]
    try:
        prepare_chess_dataset(
            name, pgn, max_positions=max_positions, per_game=per_game, nodes=nodes,
            workers=workers or None, val_fraction=val_fraction, min_elo=min_elo, seed=seed,
            endgames=endgames,
            log=lambda m: console.print(f"[dim]{m}[/]"),
        )  # fmt: skip
    except (ChessDataError, ValueError, FileExistsError) as exc:
        _fail(str(exc))


@chess_app.command("puzzles")
def puzzles(count: Annotated[int, typer.Option(help="Number of puzzles to keep")] = 2000) -> None:
    """Download Lichess puzzles (CC0) for evaluation."""
    from .evaluate import download_puzzles

    try:
        download_puzzles(count, log=console.print)
    except FileExistsError as exc:
        _fail(str(exc))


@chess_app.command("evaluate")
def evaluate(
    model: Annotated[
        str | None, typer.Argument(help="Checkpoint id, run id or path (default: best chess model)")
    ] = None,
    puzzles: Annotated[int, typer.Option(help="Puzzles to attempt (0 to skip)")] = 1000,
    games: Annotated[int, typer.Option(help="Games per Stockfish level (0 to skip)")] = 0,
    levels: Annotated[
        str, typer.Option(help="Comma-separated Stockfish UCI_Elo levels")
    ] = "1320,1600,1900",
    movetime: Annotated[float, typer.Option(help="Stockfish seconds per move")] = 0.05,
    device: Annotated[str, typer.Option()] = "auto",
    as_json: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Measure strength: held-out move quality, puzzle accuracy, games vs Stockfish."""
    from tnsr_core.checkpoints import read_meta
    from tnsr_core.evaluation import _save_result

    from .data import ChessSplit, iter_positions, resolve_chess_dataset
    from .encoding import ChessTokenizer
    from .evaluate import evaluate_puzzles, evaluation_summary, play_match, random_mover_baseline
    from .task import MOVE_EVAL_POSITIONS, move_quality

    engine = _engine(model, device)
    path = Path(engine.model_id) if Path(engine.model_id).is_absolute() else None
    from tnsr_core.paths import tnsr_home

    ckpt = path or tnsr_home() / engine.model_id
    meta = read_meta(ckpt)
    log = err.print
    t0 = time.perf_counter()
    result: dict = {
        "kind": "evaluation",
        "task": "chess",
        "checkpoint_id": engine.model_id,
        "checkpoint_name": ckpt.name,
        "step": meta.get("step"),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "device": str(engine.device),
    }
    mq = None
    ds = (meta.get("dataset") or {}).get("name")
    if ds:
        try:
            split = ChessSplit(resolve_chess_dataset(ds), "val", ChessTokenizer())
            positions = list(iter_positions(split))[:MOVE_EVAL_POSITIONS]
            mq = move_quality(engine.model, engine.tokenizer, positions, engine.head, engine.device)
            mq["positions"] = len(positions)
            log(
                f"held-out move quality: top-1 agreement {mq['move_accuracy']:.1%}, regret {mq['regret']:.4f}"
            )
        except ValueError as exc:
            log(f"skipping move quality: {exc}")
    result["move_quality"] = mq
    pz = None
    if puzzles:
        try:
            pz = evaluate_puzzles(engine, puzzles)
            log(f"puzzles: {pz['solved']}/{pz['total']} solved ({pz['accuracy']:.1%})")
        except FileNotFoundError as exc:
            log(f"skipping puzzles: {exc}")
    result["puzzles"] = pz
    result["random_baseline"] = random_mover_baseline(engine, games=6)
    match = None
    if games:
        match = play_match(engine, [int(x) for x in levels.split(",")], games, movetime, log=log)
    result["match"] = match
    result["metrics"] = evaluation_summary(pz, match, mq)
    result["seconds"] = round(time.perf_counter() - t0, 1)
    result["path"] = str(_save_result(ckpt, "evals", result))
    if as_json:
        sys.stdout.write(json.dumps(result, indent=2) + "\n")
        return
    table = Table(title=f"{engine.model_id} (step {meta.get('step')})", show_header=False)
    if mq:
        table.add_row(
            "held-out top-1 move = Stockfish",
            f"{mq['move_accuracy']:.1%} ({mq['positions']} positions)",
        )
        table.add_row("mean regret (win prob lost/move)", f"{mq['regret']:.4f}")
    if pz:
        table.add_row("puzzles solved", f"{pz['accuracy']:.1%} ({pz['solved']}/{pz['total']})")
        for k, v in pz["by_rating"].items():
            table.add_row(
                f"  puzzle rating {k}", f"{v['accuracy']:.1%} ({v['solved']}/{v['total']})"
            )
    rb = result["random_baseline"]
    table.add_row("vs random mover", f"+{rb['wins']} ={rb['draws']} -{rb['losses']}")
    if match:
        for row in match["levels"]:
            table.add_row(
                f"vs Stockfish {row['stockfish_elo']}",
                f"+{row['wins']} ={row['draws']} -{row['losses']}",
            )
        table.add_row("Elo estimate (Stockfish scale)", str(match["elo_estimate"]))
    console.print(table)
    console.print(f"[dim]saved {result['path']}[/]")


@chess_app.command("analyse")
def analyse(
    fen: Annotated[str, typer.Argument(help="FEN, or 'startpos'")] = "startpos",
    model: Annotated[
        str | None, typer.Option(help="Checkpoint (default: best chess model)")
    ] = None,
    top: Annotated[int, typer.Option()] = 8,
    device: Annotated[str, typer.Option()] = "auto",
) -> None:
    """Show the model's win probability for each legal move of a position."""
    import chess

    engine = _engine(model, device)
    board = chess.Board() if fen == "startpos" else chess.Board(fen)
    table = Table("move", "win prob (side to move)")
    for s in engine.analyse(board)[:top]:
        table.add_row(s.san, f"{s.win_prob:.3f}")
    console.print(board)
    console.print(table)


@chess_app.command("play")
def play(
    color: Annotated[str, typer.Option(help="Your colour: white or black")] = "white",
    model: Annotated[str | None, typer.Option()] = None,
    device: Annotated[str, typer.Option()] = "auto",
) -> None:
    """Play against the model in the terminal (moves in SAN or UCI, 'quit' to stop)."""
    import chess

    engine = _engine(model, device)
    board = chess.Board()
    human_white = color.lower().startswith("w")
    while not board.is_game_over():
        console.print(board.unicode(invert_color=True, borders=True))
        if (board.turn == chess.WHITE) == human_white:
            text = input("your move> ").strip()
            if text in {"quit", "exit"}:
                return
            try:
                board.push_san(text) if not (
                    len(text) >= 4 and text[1].isdigit()
                ) else board.push_uci(text)
            except ValueError:
                console.print("[red]illegal or unreadable move[/]")
        else:
            best = engine.analyse(board)[0]
            console.print(f"TNSR plays [bold]{best.san}[/] (win prob {best.win_prob:.2f})")
            board.push_uci(best.uci)
    console.print(board.unicode(invert_color=True, borders=True))
    console.print(f"Game over: {board.result()}")


@chess_app.command("uci")
def uci(
    model: Annotated[str | None, typer.Argument()] = None,
    device: Annotated[str, typer.Option()] = "auto",
) -> None:
    """Run as a UCI engine (for chess GUIs, cutechess-cli, lichess-bot)."""
    from .engine import run_uci

    run_uci(lambda: _engine(model, device))

import random
import shutil

import chess
import numpy as np
import pytest
import torch

from tnsr_core.chess.data import (
    ChessDataError,
    ChessSplit,
    build_dataset_from_labels,
    cp_to_win_prob,
    game_split,
    iter_positions,
    prepare_chess_dataset,
    resolve_chess_dataset,
)
from tnsr_core.chess.encoding import (
    SEQ_LEN,
    STATE_LEN,
    ChessTokenizer,
    all_actions,
    bin_centers,
    encode_pairs,
    win_prob_to_bin,
)
from tnsr_core.chess.engine import ChessEngine, run_uci, win_prob_to_cp
from tnsr_core.chess.evaluate import expected_score, performance_rating, solve_puzzle
from tnsr_core.config import load_config
from tnsr_core.training import Trainer

VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0,
}


def material_label(board: chess.Board) -> list[tuple[str, float]]:
    """Cheap stand-in for Stockfish: win prob from material balance after each move."""
    side = board.turn
    out = []
    for m in board.legal_moves:
        board.push(m)
        diff = sum(
            VALUES[p.piece_type] * (1 if p.color == side else -1)
            for p in board.piece_map().values()
        )
        board.pop()
        out.append((m.uci(), cp_to_win_prob(100 * diff)))
    return out


def random_positions(n: int, seed: int = 0) -> list[chess.Board]:
    rng = random.Random(seed)
    boards = []
    while len(boards) < n:
        b = chess.Board()
        for _ in range(rng.randint(4, 40)):
            if b.is_game_over():
                break
            b.push(rng.choice(list(b.legal_moves)))
        if not b.is_game_over():
            boards.append(b)
    return boards


@pytest.fixture
def chess_dataset(home):
    items = []
    for i, b in enumerate(random_positions(240)):
        items.append((b.fen(), material_label(b), "val" if i % 6 == 0 else "train"))
    return build_dataset_from_labels(
        "chess-demo", items, source="random-games", labeling="material"
    )


def test_action_vocabulary_covers_all_legal_moves():
    tok = ChessTokenizer()
    assert len(all_actions()) == 1968 and tok.num_actions == 1968
    assert len(set(all_actions())) == 1968
    for b in random_positions(200, seed=1):
        for m in b.legal_moves:
            assert tok.action_from_index(tok.action_index(m.uci())) == m.uci()
    with pytest.raises(ValueError):
        tok.action_id("a1a1")


def test_position_encoding_is_fixed_length_and_distinguishes_positions():
    tok = ChessTokenizer()
    start = chess.Board()
    enc = tok.encode_fen(start.fen())
    assert (
        enc.shape == (STATE_LEN,) and enc.min() >= tok.char_offset and enc.max() < tok.action_offset
    )
    assert tok.fen_to_chars(start.fen()).startswith("wrnbqkbnr")
    b = start.copy()
    b.push_uci("e2e4")
    assert not np.array_equal(tok.encode_fen(b.fen()), enc)
    # side to move, castling, en passant and clocks are all encoded
    b2 = chess.Board("r3k2r/8/8/3pP3/8/8/8/R3K2R w Kq d6 12 140")
    chars = tok.fen_to_chars(b2.fen())
    assert (
        len(chars) == STATE_LEN
        and chars[65:69] == "Kq.."
        and chars[69:71] == "d6"
        and chars[71:74] == "12."
    )
    pairs = encode_pairs(tok, start.fen(), ["e2e4", "g1f3"])
    assert pairs.shape == (2, SEQ_LEN) and pairs[0, -1] == tok.action_id("e2e4")


def test_tokenizer_serialization_roundtrip():
    from tnsr_core.tokenizer import tokenizer_from_dict

    tok = ChessTokenizer()
    again = tokenizer_from_dict(tok.to_dict())
    assert isinstance(again, ChessTokenizer) and again.identity == tok.identity


def test_bins_and_score_conversions():
    assert win_prob_to_bin(0.0, 128) == 0 and win_prob_to_bin(1.0, 128) == 127
    assert win_prob_to_bin(0.5, 4) == 2
    assert np.allclose(bin_centers(4), [0.125, 0.375, 0.625, 0.875])
    assert cp_to_win_prob(0) == 0.5 and cp_to_win_prob(300) > 0.7
    for cp in (-500, -50, 0, 120, 900):
        assert abs(win_prob_to_cp(cp_to_win_prob(cp)) - cp) <= 1


def test_split_is_by_game_and_deterministic():
    assert game_split("https://lichess.org/abc", 0.5, 0) == game_split(
        "https://lichess.org/abc", 0.5, 0
    )
    splits = {game_split(f"g{i}", 0.2, 0) for i in range(200)}
    assert splits == {"train", "val"}


def test_dataset_loading_and_batches(chess_dataset):
    tok = ChessTokenizer()
    path = resolve_chess_dataset("chess-demo")
    train = ChessSplit(path, "train", tok)
    x, y, wp = train.batch(torch.arange(16), 32)
    assert x.shape == (16, SEQ_LEN) and y.max() < 32 and ((wp >= 0) & (wp <= 1)).all()
    assert (x[:, -1] >= tok.action_offset).all()
    fen, moves = next(iter_positions(train))
    assert len(moves) == chess.Board(fen).legal_moves.count()
    with pytest.raises(ChessDataError):
        resolve_chess_dataset("nope")


def test_chess_training_learns_material_and_engine_plays_legal_moves(chess_dataset):
    cfg = load_config(
        "chess-tiny",
        {
            "train.max_steps": 150,
            "train.eval_interval": 75,
            "train.learning_rate": 3e-3,
            "model.n_embd": 64,
            "model.n_head": 4,
        },
    )
    result = Trainer.create(cfg, log=lambda _m: None).train()
    assert result.status == "completed"
    metrics = [
        m
        for m in __import__("tnsr_core.training", fromlist=["read_metrics"]).read_metrics(
            result.run_dir
        )
        if m["type"] == "eval"
    ]
    assert "move_accuracy" in metrics[-1] and "regret" in metrics[-1]
    assert metrics[-1]["val_loss"] < metrics[0]["val_loss"] or metrics[-1]["regret"] < 0.2

    engine = ChessEngine.from_checkpoint(result.last_checkpoint, "cpu")
    board = chess.Board()
    for _ in range(40):  # self-play: only legal moves, never crashes
        if board.is_game_over():
            break
        board.push(engine.best_move(board))
    # analyse() scores every legal move and returns them best-first
    hanging = chess.Board("4k3/8/8/3q4/8/8/8/3QK3 w - - 0 1")
    scores = engine.analyse(hanging)
    assert len(scores) == hanging.legal_moves.count()
    assert scores[0].win_prob >= scores[-1].win_prob


def test_resume_chess_run(chess_dataset):
    cfg = load_config("chess-tiny", {"train.max_steps": 20})
    first = Trainer.create(cfg, log=lambda _m: None)
    orig = first.optimizer.step

    def stop_after(*a, **kw):
        out = orig(*a, **kw)
        if first.step + 1 >= 10:
            first.stop.request("test")
        return out

    first.optimizer.step = stop_after
    assert first.train().status == "interrupted"
    resumed = Trainer.resume_run(first.run_dir, log=lambda _m: None)
    assert resumed.step == 10 and resumed.train().status == "completed"


def test_uci_protocol(chess_dataset):
    cfg = load_config("chess-tiny", {"train.max_steps": 5, "train.eval_interval": 5})
    ckpt = Trainer.create(cfg, log=lambda _m: None).train().last_checkpoint
    import io

    out = io.StringIO()
    cmds = [
        "uci",
        "isready",
        "position startpos moves e2e4 e7e5",
        "go movetime 100",
        "position fen 7k/8/8/8/8/8/8/K7 w - - 0 1",
        "go",
        "quit",
    ]
    run_uci(lambda: ChessEngine.from_checkpoint(ckpt, "cpu"), cmds, out)
    lines = out.getvalue().splitlines()
    assert "uciok" in lines and "readyok" in lines
    best = [line.split()[1] for line in lines if line.startswith("bestmove")]
    assert len(best) == 2
    b = chess.Board()
    b.push_uci("e2e4")
    b.push_uci("e7e5")
    assert chess.Move.from_uci(best[0]) in b.legal_moves


def test_puzzle_solving_logic():
    class Oracle:
        def __init__(self, line):
            self.line = line

        def best_move(self, board):
            return chess.Move.from_uci(self.line.pop(0))

    fen = "r6k/pp2r2p/4Rp1Q/3p4/8/1N1P2R1/PqP2bPP/7K b - - 0 24"
    moves = ["f2g3", "e6e7", "b2b1", "b3c1", "b1c1", "h6c1"]
    assert solve_puzzle(Oracle(["e6e7", "b3c1", "h6c1"]), fen, moves)
    assert not solve_puzzle(Oracle(["h6h7", "b3c1", "h6c1"]), fen, moves)
    # any mating move is accepted when the solution mates
    mate_fen = "6k1/5ppp/8/8/8/8/5PPP/R5K1 b - - 0 1"
    assert solve_puzzle(Oracle(["a1a8"]), mate_fen, ["g8h8", "a1a8"])


def test_performance_rating():
    assert performance_rating([(1500, 0.5), (1500, 0.5)]) == pytest.approx(1500, abs=1)
    assert performance_rating([(1500, 1.0), (1500, 0.0), (1500, 1.0)]) > 1500
    assert performance_rating([(1500, 1.0)]) is None and performance_rating([]) is None
    r = performance_rating([(1320, 0.75)] * 4)
    assert expected_score(r, 1320) == pytest.approx(0.75, abs=1e-3)


def test_non_chess_checkpoint_rejected(trained):
    from tnsr_core.checkpoints import CheckpointError

    with pytest.raises(CheckpointError, match="not a TNSR chess checkpoint"):
        ChessEngine.from_checkpoint(trained, "cpu")


@pytest.mark.skipif(
    shutil.which("stockfish") is None
    and not __import__("pathlib").Path("/usr/games/stockfish").exists(),
    reason="Stockfish not installed",
)
def test_stockfish_labeling_pipeline(home, tmp_path):
    pgn = tmp_path / "games.pgn"
    pgn.write_text(
        '[Event "t"]\n[Site "https://lichess.org/g1"]\n[Result "1-0"]\n\n1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 1-0\n\n'
        '[Event "t"]\n[Site "https://lichess.org/g2"]\n[Result "0-1"]\n\n1. d4 d5 2. c4 e6 3. Nc3 Nf6 4. Bg5 Be7 0-1\n\n'
    )
    path = prepare_chess_dataset(
        "sf",
        pgn,
        max_positions=6,
        per_game=3,
        nodes=200,
        workers=1,
        val_fraction=0.3,
        log=lambda _m: None,
    )
    meta = __import__("json").loads((path / "meta.json").read_text())
    assert meta["complete"] and meta["labeling"]["engine"].lower().startswith("stockfish")
    total = meta["stats"]["train_positions"] + meta["stats"]["val_positions"]
    assert 1 <= total <= 6
    # re-running resumes instead of failing or relabeling
    again = prepare_chess_dataset(
        "sf",
        pgn,
        max_positions=6,
        per_game=3,
        nodes=200,
        workers=1,
        val_fraction=0.3,
        log=lambda _m: None,
    )
    assert again == path


def test_hl_gauss_targets_are_centred_distributions():
    from tnsr_core.chess.encoding import hl_gauss_targets

    h = hl_gauss_targets(np.array([0.0, 0.5, 1.0, 0.3]), 128)
    assert h.shape == (4, 128) and np.allclose(h.sum(1), 1)
    assert h[0].argmax() == 0 and h[2].argmax() == 127 and h[3].argmax() == int(0.3 * 128)
    assert h[1, 63] == pytest.approx(h[1, 64])  # 0.5 sits exactly on the edge between two bins
    assert (h >= 0).all() and h[3, 38] > h[3, 30]


def test_move_marks_write_the_move_onto_the_board():
    from tnsr_core.chess.encoding import SQUARE_CHARS, board_position
    from tnsr_core.tokenizer import tokenizer_from_dict

    tok = ChessTokenizer()  # marks on by default
    plain = ChessTokenizer(move_marks=False)
    assert tok.vocab_size == plain.vocab_size + 2 * len(SQUARE_CHARS)
    pairs = encode_pairs(tok, chess.STARTING_FEN, ["e2e4", "g1f3"])
    e2, e4, g1, f3 = (board_position(s) for s in ("e2", "e4", "g1", "f3"))
    assert pairs[0, e2] == tok.from_offset + SQUARE_CHARS.index("P")
    assert pairs[0, e4] == tok.to_offset + SQUARE_CHARS.index(".")
    assert (
        pairs[1, g1] == tok.from_offset + SQUARE_CHARS.index("N") and pairs[1, e2] != pairs[0, e2]
    )
    # everything else is the plain position encoding
    base = plain.encode_fen(chess.STARTING_FEN)
    untouched = [i for i in range(STATE_LEN) if i not in (e2, e4)]
    assert (pairs[0, untouched] == base[untouched]).all()
    assert pairs[0, -1] == tok.action_id("e2e4")
    # serialization keeps the flag; the unmarked layout keeps its v1 identity
    assert tokenizer_from_dict(tok.to_dict()).move_marks is True
    assert tokenizer_from_dict(plain.to_dict()).move_marks is False
    assert plain.to_dict()["version"] == 1 and "move_marks" not in plain.to_dict()


def test_batches_apply_marks_consistently_with_inference(chess_dataset):
    tok = ChessTokenizer()
    split = ChessSplit(resolve_chess_dataset("chess-demo"), "val", tok)
    fen, moves = next(iter_positions(split))
    x, _, _ = split.batch(torch.arange(len(moves)), 16)
    expected = encode_pairs(tok, fen, [u for u, _ in moves])
    assert torch.equal(x, torch.from_numpy(expected))


def test_value_head_mate_classes_rank_faster_mates_first():
    from tnsr_core.chess.encoding import ValueHead

    head = ValueHead(value_bins=8, mate_classes=3)
    assert head.num_classes == 14
    cls = head.classes(np.array([0.5, 1.0, 1.0, 0.0, 0.0, 0.99]), np.array([0, 1, 99, -1, -3, 0]))
    assert cls.tolist() == [4, 8, 10, 11, 13, 7]
    r = head.rank_values
    # mate in 1 > mate in 2 > mate in 3 > best ordinary bin; being mated slowly > being mated fast
    assert r[8] > r[9] > r[10] > r[7] and r[13] > r[12] > r[11] and r[11] < r[0]
    assert head.win_values[8] == 1.0 and head.win_values[11] == 0.0
    soft = head.soft_targets(np.array([0.5, 1.0]), np.array([0, 2]))
    assert np.allclose(soft.sum(1), 1) and soft[1, 9] == 1.0 and soft[0, 8:].sum() == 0
    assert ValueHead.from_meta({}, 128).mate_classes == 0  # checkpoints from before mate classes
    with pytest.raises(ValueError):
        ValueHead.from_meta({"chess": {"value_bins": 128, "mate_classes": 20}}, 128)


def test_checkpoint_records_value_head(chess_dataset):
    cfg = load_config("chess-tiny", {"train.max_steps": 4, "train.eval_interval": 4})
    ckpt = Trainer.create(cfg, log=lambda _m: None).train().last_checkpoint
    meta = __import__("json").loads((ckpt / "meta.json").read_text())
    assert meta["chess"] == {"value_bins": 32, "mate_classes": 4}
    engine = ChessEngine.from_checkpoint(ckpt, "cpu")
    assert engine.head.num_classes == 40
    scores = engine.analyse(chess.Board())
    assert all(0.0 <= s.win_prob <= 1.0 for s in scores)


def test_endgame_generator_is_legal_and_deterministic():
    from tnsr_core.chess.endgames import generate_endgames

    a = generate_endgames(50, seed=3)
    assert a == generate_endgames(50, seed=3) and len(a) == 50 and len(set(a)) == 50
    for fen in a:
        b = chess.Board(fen)
        assert b.is_valid() and not b.is_game_over() and len(b.piece_map()) <= 7


def test_rules_scoring_takes_mate_and_avoids_stalemate(chess_dataset):
    from tnsr_core.chess.engine import move_result

    cfg = load_config("chess-tiny", {"train.max_steps": 5, "train.eval_interval": 5})
    ckpt = Trainer.create(cfg, log=lambda _m: None).train().last_checkpoint
    # Qh7# mates; Qg6 (among others) stalemates the bare king.
    board = chess.Board("7k/8/5K2/8/8/8/8/6Q1 w - - 0 1")
    assert move_result(board, chess.Move.from_uci("g1g7")) == 1.0
    assert move_result(board, chess.Move.from_uci("g1g6")) == 0.5
    assert move_result(board, chess.Move.from_uci("g1a1")) is None
    assert board.fen() == "7k/8/5K2/8/8/8/8/6Q1 w - - 0 1"  # board left untouched

    plain = ChessEngine.from_checkpoint(ckpt, "cpu")
    ruled = ChessEngine.from_checkpoint(ckpt, "cpu", rules=True)
    scores = {s.uci: s.win_prob for s in ruled.analyse(board)}
    assert scores["g1g7"] == 1.0 and scores["g1g6"] == 0.5
    best = ruled.best_move(board)
    board.push(best)
    assert board.is_checkmate()
    assert len(plain.analyse(chess.Board())) == 20

    # A move that allows a threefold-repetition claim is a draw, which needs the game history.
    hist = chess.Board("7k/8/8/8/8/8/R7/K7 w - - 0 1")
    for uci in ["a2b2", "h8g8", "b2a2", "g8h8", "a2b2", "h8g8", "b2a2"]:
        hist.push_uci(uci)
    assert move_result(hist, chess.Move.from_uci("g8h8")) == 0.5


def test_combined_datasets_train_together(home):
    items_a = [
        (b.fen(), material_label(b), "val" if i % 6 == 0 else "train")
        for i, b in enumerate(random_positions(60, seed=1))
    ]
    items_b = [
        (b.fen(), material_label(b), "val" if i % 6 == 0 else "train")
        for i, b in enumerate(random_positions(60, seed=2))
    ]
    pa = build_dataset_from_labels("chess-a", items_a, source="a", labeling="material")
    pb = build_dataset_from_labels("chess-b", items_b, source="b", labeling="material")
    tok = ChessTokenizer()
    both = ChessSplit([pa, pb], "train", tok)
    assert both.num_records == (
        ChessSplit(pa, "train", tok).num_records + ChessSplit(pb, "train", tok).num_records
    )
    assert both.fens is not None and both.num_positions == len(both.fens)

    cfg = load_config("chess-tiny", {"data.dataset": "chess-a+chess-b", "train.max_steps": 4})
    result = Trainer.create(cfg, log=lambda _m: None).train()
    assert result.status == "completed"
    import json

    ref = json.loads((result.run_dir / "run.json").read_text())["dataset"]
    assert ref["name"] == "chess-a+chess-b"
    assert ref["identity"].count("+") == 1
    assert len(ref["sources"]) == 2
    with pytest.raises(ChessDataError):
        Trainer.create(
            load_config("chess-tiny", {"data.dataset": "chess-a+missing"}), log=lambda _m: None
        )


def test_collect_games_writes_labelable_pgn(chess_dataset, tmp_path):
    import chess.pgn

    from tnsr_core.chess.selfplay import collect_games

    cfg = load_config("chess-tiny", {"train.max_steps": 5, "train.eval_interval": 5})
    ckpt = Trainer.create(cfg, log=lambda _m: None).train().last_checkpoint
    engine = ChessEngine.from_checkpoint(ckpt, "cpu")
    out = tmp_path / "tnsr-test.pgn"
    collect_games(engine, out, random_games=3, max_plies=40, log=lambda _m: None)
    with out.open() as fh:
        games = [g for g in iter(lambda: chess.pgn.read_game(fh), None)]
    assert len(games) == 3
    assert len({g.headers["Site"] for g in games}) == 3
    for g in games:
        assert not g.errors and len(list(g.mainline_moves())) >= 4
    with pytest.raises(FileExistsError):
        collect_games(engine, out, random_games=1, log=lambda _m: None)


def _bag_bytes(records: list[tuple[str, str, float]]) -> bytes:
    import struct

    def varint(n: int) -> bytes:
        out = bytearray()
        while True:
            b = n & 0x7F
            n >>= 7
            out.append(b | (0x80 if n else 0))
            if not n:
                return bytes(out)

    body, limits = b"", []
    for fen, move, wp in records:
        rec = varint(len(fen)) + fen.encode() + varint(len(move)) + move.encode()
        body += rec + struct.pack(">d", wp)
        limits.append(len(body))
    return body + b"".join(x.to_bytes(8, "little") for x in limits)


def test_chessbench_bag_import(home, tmp_path):
    from tnsr_core.chess.chessbench import import_chessbench, read_bag

    boards = random_positions(80, seed=3)
    records = [(b.fen(), uci, wp) for b in boards for uci, wp in material_label(b)]
    bag = tmp_path / "action_value-00000-of-02148_data.bag"
    bag.write_bytes(_bag_bytes(records))
    back = list(read_bag(bag))
    assert back == [(f, m, pytest.approx(w)) for f, m, w in records]

    excluded = {boards[0].fen()}
    path = import_chessbench(
        "chessbench-test", [bag], exclude_fens=excluded, val_fraction=0.2, log=lambda _m: None
    )
    tok = ChessTokenizer()
    train, val = ChessSplit(path, "train", tok), ChessSplit(path, "val", tok)
    n_excluded = sum(1 for r in records if r[0] == boards[0].fen())
    assert train.num_records + val.num_records == len(records) - n_excluded
    assert boards[0].fen() not in (train.fens or []) + (val.fens or [])
    assert int((train.mates != 0).sum()) == 0  # ChessBench has no mate distances
    import json

    meta = json.loads((path / "meta.json").read_text())
    assert (
        "CC-BY" in meta["source"]["license"] and meta["sampling"]["records_excluded"] == n_excluded
    )
    with pytest.raises(FileExistsError):
        import_chessbench("chessbench-test", [bag], log=lambda _m: None)
    bad = tmp_path / "bad.bag"
    bad.write_bytes(b"\x00" * 4)
    with pytest.raises(ChessDataError):
        list(read_bag(bad))


def test_shared_board_queries_are_independent():
    from tnsr_core.config import ModelConfig
    from tnsr_core.model import TNSRModel

    tok = ChessTokenizer(move_marks=False)
    cfg = ModelConfig(
        vocab_size=tok.vocab_size, block_size=STATE_LEN + 1, n_layer=2, n_head=2, n_embd=32,
        causal=False, num_classes=10, query_prefix=STATE_LEN,
    )  # fmt: skip
    torch.manual_seed(0)
    model = TNSRModel(cfg).eval()
    board = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3")
    ucis = [m.uci() for m in board.legal_moves]
    from tnsr_core.chess.encoding import encode_shared

    with torch.no_grad():
        together, _ = model(torch.from_numpy(encode_shared(tok, board.fen(), ucis)))
        for i in (0, 5, len(ucis) - 1):
            alone, _ = model(torch.from_numpy(encode_shared(tok, board.fen(), [ucis[i]])))
            assert torch.allclose(together[0, i], alone[0, 0], atol=1e-5)
        # Reordering the moves only reorders the outputs.
        rev, _ = model(torch.from_numpy(encode_shared(tok, board.fen(), ucis[::-1])))
        assert torch.allclose(together[0], rev[0].flip(0), atol=1e-5)
    with pytest.raises(ValueError):
        encode_shared(ChessTokenizer(), board.fen(), ucis)  # marks need one sequence per move


def test_shared_board_training_and_engine(chess_dataset):
    cfg = load_config(
        "chess-tiny",
        {"data.shared_board": True, "data.move_marks": False, "train.max_steps": 40,
         "train.batch_size": 16, "train.eval_interval": 20},
    )  # fmt: skip
    result = Trainer.create(cfg, log=lambda _m: None).train()
    assert result.status == "completed"
    from tnsr_core.training import read_metrics

    evals = [m for m in read_metrics(result.run_dir) if m["type"] == "eval"]
    assert "move_accuracy" in evals[-1] and evals[-1]["val_loss"] < evals[0]["val_loss"] + 0.5
    engine = ChessEngine.from_checkpoint(result.last_checkpoint, "cpu")
    assert engine.model.config.query_prefix == STATE_LEN
    board = chess.Board()
    for _ in range(20):
        if board.is_game_over():
            break
        board.push(engine.best_move(board))
    scores = engine.analyse(chess.Board())
    assert len(scores) == 20 and scores[0].win_prob >= scores[-1].win_prob

    soft = load_config(
        "chess-tiny",
        {"data.shared_board": True, "data.move_marks": False, "train.max_steps": 5,
         "train.batch_size": 8, "train.value_loss": "hl_gauss", "train.eval_interval": 5},
    )  # fmt: skip
    assert Trainer.create(soft, log=lambda _m: None).train().status == "completed"
    with pytest.raises(ValueError):
        Trainer.create(load_config("chess-tiny", {"data.shared_board": True}), log=lambda _m: None)


def test_position_batch_matches_records(chess_dataset):
    from tnsr_core.chess.encoding import ValueHead

    tok = ChessTokenizer(move_marks=False)
    split = ChessSplit(chess_dataset, "train", tok)
    head = ValueHead(32, 4)
    pos = torch.tensor([0, 3, 7])
    x, y = split.position_batch(pos, head)
    for row, p in enumerate(pos.tolist()):
        lo, hi = int(split.pos_offsets[p]), int(split.pos_offsets[p + 1])
        assert torch.equal(x[row, :STATE_LEN], split.states[p].long())
        n = hi - lo
        assert torch.equal(
            x[row, STATE_LEN : STATE_LEN + n], split.actions[lo:hi] + tok.action_offset
        )
        xr, yr, _ = split.batch(torch.arange(lo, hi), head)
        assert torch.equal(y[row, :n], yr) and bool((y[row, n:] == -100).all())
        assert bool((x[row, STATE_LEN + n :] == 0).all())

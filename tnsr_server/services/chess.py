"""Games against the TNSR chess engine. The server is the rules authority: the UI sends
moves, the server validates them with python-chess, replies with the engine and persists."""

from __future__ import annotations

import threading
from typing import Any

import chess

from tnsr_core.checkpoints import (
    CheckpointError,
    checkpoint_id,
    discover_checkpoints,
    preferred_checkpoint,
    resolve_checkpoint_id,
)
from tnsr_core.chess.engine import ChessEngine
from tnsr_server.persistence import Database

TOP_MOVES = 8


class ChessError(ValueError):
    pass


class ChessService:
    def __init__(self, db: Database, device: str = "auto"):
        self.db = db
        self.device = device
        self._engines: dict[str, ChessEngine] = {}
        self._lock = threading.Lock()

    # -- models ------------------------------------------------------------------------

    def list_models(self) -> list[dict[str, Any]]:
        models = [
            c for c in discover_checkpoints() if (c.get("tokenizer") or {}).get("kind") == "chess"
        ]
        preferred = preferred_checkpoint(models)
        for m in models:
            m["recommended"] = preferred is not None and m["id"] == preferred["id"]
            m["loaded"] = m["id"] in self._engines
        return models

    def default_model(self) -> str:
        choice = preferred_checkpoint(self.list_models())
        if choice is None:
            raise ChessError(
                "No chess models found. Train one first: tnsr train --config chess-mini"
            )
        return choice["id"]

    def engine(self, model_id: str | None) -> ChessEngine:
        model_id = model_id or self.default_model()
        try:
            path = resolve_checkpoint_id(model_id)
        except (CheckpointError, ValueError) as exc:
            raise ChessError(f"Model unavailable: {exc}") from exc
        key = checkpoint_id(path)
        with self._lock:
            if key not in self._engines:
                try:
                    engine = ChessEngine.from_checkpoint(path, self.device)
                except CheckpointError as exc:
                    raise ChessError(str(exc)) from exc
                self._engines = {key: engine}  # keep one engine in memory
            return self._engines[key]

    # -- positions -----------------------------------------------------------------------

    @staticmethod
    def board_for(game: dict[str, Any]) -> chess.Board:
        board = chess.Board(game["start_fen"])
        for uci in game["moves"]:
            board.push_uci(uci)
        return board

    @staticmethod
    def describe(board: chess.Board) -> dict[str, Any]:
        replay = board.root()
        san = []
        for move in board.move_stack:
            san.append(replay.san(move))
            replay.push(move)
        outcome = board.outcome(claim_draw=True)
        return {
            "fen": board.fen(),
            "turn": "white" if board.turn == chess.WHITE else "black",
            "moves": [m.uci() for m in board.move_stack],
            "san": san,
            "legal": [m.uci() for m in board.legal_moves] if outcome is None else [],
            "check": board.is_check(),
            "last_move": board.move_stack[-1].uci() if board.move_stack else None,
            "outcome": None
            if outcome is None
            else {"result": outcome.result(), "termination": outcome.termination.name.lower()},
        }

    def analyse(self, board: chess.Board, model_id: str | None) -> dict[str, Any]:
        engine = self.engine(model_id)
        with self._lock:
            scores = engine.analyse(board)
        return {
            "model_id": engine.model_id,
            "moves": [
                {"uci": s.uci, "san": s.san, "win_prob": round(s.win_prob, 4)}
                for s in scores[:TOP_MOVES]
            ],
            "legal_count": len(scores),
        }

    # -- games ---------------------------------------------------------------------------

    def _view(
        self, game: dict[str, Any], engine_move: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        board = self.board_for(game)
        view = {**game, "state": self.describe(board), "engine_move": engine_move, "analysis": None}
        if view["state"]["outcome"] is None:
            try:
                view["analysis"] = self.analyse(board, game["model_id"])
            except ChessError:
                view["analysis"] = None
        return view

    def get(self, gid: str) -> dict[str, Any]:
        game = self.db.get_game(gid)
        if game is None:
            raise KeyError(gid)
        return self._view(game)

    def create(
        self, human_color: str, model_id: str | None, start_fen: str | None
    ) -> dict[str, Any]:
        model_id = model_id or self.default_model()
        self.engine(model_id)  # fail fast if the model cannot be loaded
        fen = start_fen or chess.STARTING_FEN
        try:
            board = chess.Board(fen)
        except ValueError as exc:
            raise ChessError(f"Invalid FEN: {exc}") from exc
        if not board.is_valid():
            raise ChessError("Illegal position")
        title = "TNSR (white) vs you" if human_color == "black" else "You (white) vs TNSR"
        game = self.db.create_game(title, model_id, human_color, board.fen())
        engine_move = None
        if (board.turn == chess.WHITE) != (human_color == "white"):
            game, engine_move = self._engine_reply(game)
        return self._view(game, engine_move)

    def _engine_reply(self, game: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | None]:
        board = self.board_for(game)
        if board.outcome(claim_draw=True) is not None:
            return game, None
        engine = self.engine(game["model_id"])
        with self._lock:
            scores = engine.analyse(board)
        best = scores[0]
        board.push_uci(best.uci)
        outcome = board.outcome(claim_draw=True)
        updated = self.db.update_game(
            game["id"],
            moves=[*game["moves"], best.uci],
            result=outcome.result() if outcome else None,
        )
        assert updated is not None
        info = {
            "uci": best.uci,
            "san": best.san,
            "win_prob": round(best.win_prob, 4),
            "alternatives": [
                {"uci": s.uci, "san": s.san, "win_prob": round(s.win_prob, 4)}
                for s in scores[:TOP_MOVES]
            ],
        }
        return updated, info

    def move(self, gid: str, uci: str) -> dict[str, Any]:
        game = self.db.get_game(gid)
        if game is None:
            raise KeyError(gid)
        board = self.board_for(game)
        if board.outcome(claim_draw=True) is not None:
            raise ChessError("The game is over")
        human_white = game["human_color"] == "white"
        if (board.turn == chess.WHITE) != human_white:
            raise ChessError("It is TNSR's turn")
        try:
            move = chess.Move.from_uci(uci)
        except ValueError as exc:
            raise ChessError(f"Not a UCI move: {uci!r}") from exc
        if move not in board.legal_moves:
            raise ChessError(f"Illegal move {uci} in this position")
        board.push(move)
        outcome = board.outcome(claim_draw=True)
        game = (
            self.db.update_game(
                gid, moves=[*game["moves"], uci], result=outcome.result() if outcome else None
            )
            or game
        )
        engine_move = None
        if outcome is None:
            game, engine_move = self._engine_reply(game)
        return self._view(game, engine_move)

    def undo(self, gid: str) -> dict[str, Any]:
        """Take back moves until it is the human's turn again (at least one human move)."""
        game = self.db.get_game(gid)
        if game is None:
            raise KeyError(gid)
        moves = list(game["moves"])
        if not moves:
            raise ChessError("Nothing to undo")
        board = chess.Board(game["start_fen"])
        human_white = game["human_color"] == "white"
        human_first = (board.turn == chess.WHITE) == human_white
        while moves:
            moves.pop()
            # after popping, it must be the human's turn and at least one human move removed
            human_turn = (len(moves) % 2 == 0) == human_first
            if human_turn:
                break
        if not moves and not human_first:
            raise ChessError("Nothing to undo")
        updated = self.db.update_game(gid, moves=moves, result=None)
        assert updated is not None
        return self._view(updated)

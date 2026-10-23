"""Chess endpoints for TNSR Play."""

from __future__ import annotations

from typing import Any

import chess
from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool

from tnsr_server.context import ServerContext, get_ctx
from tnsr_server.services.chess import ChessError

from .schemas import AnalyseRequest, GameCreate, GameUpdate, MoveRequest

router = APIRouter(prefix="/api/chess")


async def _call(fn, *args):
    try:
        return await run_in_threadpool(fn, *args)
    except KeyError as exc:
        raise HTTPException(404, "Game not found") from exc
    except ChessError as exc:
        raise HTTPException(409, str(exc)) from exc


@router.get("/models")
async def chess_models(ctx: ServerContext = Depends(get_ctx)) -> dict[str, Any]:
    return {"models": await run_in_threadpool(ctx.chess.list_models)}


@router.get("/games")
def list_games(ctx: ServerContext = Depends(get_ctx)) -> dict[str, Any]:
    return {"games": ctx.db.list_games()}


@router.post("/games", status_code=201)
async def create_game(body: GameCreate, ctx: ServerContext = Depends(get_ctx)) -> dict[str, Any]:
    return await _call(ctx.chess.create, body.human_color, body.model_id, body.start_fen)


@router.get("/games/{gid}")
async def get_game(gid: str, ctx: ServerContext = Depends(get_ctx)) -> dict[str, Any]:
    return await _call(ctx.chess.get, gid)


@router.patch("/games/{gid}")
def rename_game(
    gid: str, body: GameUpdate, ctx: ServerContext = Depends(get_ctx)
) -> dict[str, Any]:
    game = ctx.db.update_game(gid, title=body.title.strip())
    if game is None:
        raise HTTPException(404, "Game not found")
    return game


@router.delete("/games/{gid}", status_code=204)
def delete_game(gid: str, ctx: ServerContext = Depends(get_ctx)) -> None:
    if not ctx.db.delete_game(gid):
        raise HTTPException(404, "Game not found")


@router.post("/games/{gid}/move")
async def play_move(
    gid: str, body: MoveRequest, ctx: ServerContext = Depends(get_ctx)
) -> dict[str, Any]:
    return await _call(ctx.chess.move, gid, body.move)


@router.post("/games/{gid}/undo")
async def undo_move(gid: str, ctx: ServerContext = Depends(get_ctx)) -> dict[str, Any]:
    return await _call(ctx.chess.undo, gid)


@router.post("/analyse")
async def analyse(body: AnalyseRequest, ctx: ServerContext = Depends(get_ctx)) -> dict[str, Any]:
    try:
        board = chess.Board(body.fen)
    except ValueError as exc:
        raise HTTPException(422, f"Invalid FEN: {exc}") from exc
    return {
        "state": ctx.chess.describe(board),
        "analysis": await _call(ctx.chess.analyse, board, body.model_id),
    }

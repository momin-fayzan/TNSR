"""FastAPI application factory."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from tnsr_core import __version__
from tnsr_core.paths import tnsr_home, ui_dist_dir, var_dir
from tnsr_server.api.chess_routes import router as chess_router
from tnsr_server.api.routes import router
from tnsr_server.context import ServerContext
from tnsr_server.persistence import Database
from tnsr_server.services.chess import ChessService
from tnsr_server.services.experiments import ExperimentService
from tnsr_server.services.generation import GenerationService
from tnsr_server.services.jobs import JobManager
from tnsr_server.services.models import ModelManager

log = logging.getLogger("tnsr.server")

DEV_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"]


def create_app(
    db_path: Path | str | None = None, device: str | None = None, serve_ui: bool = True
) -> FastAPI:
    home = tnsr_home()
    db = Database(db_path or os.environ.get("TNSR_DB") or var_dir() / "tnsr.db")
    recovered = db.recover_interrupted_messages()
    if recovered:
        log.warning("Marked %d interrupted message(s) from a previous session", recovered)
    models = ModelManager(device or os.environ.get("TNSR_DEVICE", "auto"))
    ctx = ServerContext(
        home=home,
        db=db,
        models=models,
        generation=GenerationService(models),
        jobs=JobManager(db, home),
        experiments=ExperimentService(db),
        chess=ChessService(db, device or os.environ.get("TNSR_DEVICE", "auto")),
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        ctx.jobs.shutdown()
        db.close()

    app = FastAPI(title="TNSR", version=__version__, lifespan=lifespan)
    app.state.ctx = ctx
    app.add_middleware(
        CORSMiddleware,
        allow_origins=DEV_ORIGINS,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(Exception)
    async def unhandled(_request: Request, exc: Exception):
        log.exception("Unhandled error")
        return JSONResponse(
            status_code=500, content={"detail": f"Internal error: {type(exc).__name__}: {exc}"}
        )

    app.include_router(router)
    app.include_router(chess_router)

    dist = ui_dist_dir()
    if serve_ui and (dist / "index.html").is_file():
        assets = dist / "assets"
        if assets.is_dir():
            app.mount("/assets", StaticFiles(directory=assets), name="assets")

        @app.get("/{full_path:path}", include_in_schema=False)
        async def spa(full_path: str):
            if full_path.startswith("api/"):
                return JSONResponse(status_code=404, content={"detail": "Not found"})
            candidate = (dist / full_path).resolve()
            if full_path and candidate.is_file() and dist in candidate.parents:
                return FileResponse(candidate)
            return FileResponse(dist / "index.html")

    return app

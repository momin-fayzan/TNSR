"""Server-wide state shared by the API routes."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import Request

from tnsr_server.persistence import Database
from tnsr_server.services.chess import ChessService
from tnsr_server.services.experiments import ExperimentService
from tnsr_server.services.generation import GenerationService
from tnsr_server.services.jobs import JobManager
from tnsr_server.services.models import ModelManager


@dataclass
class ServerContext:
    home: Path
    db: Database
    models: ModelManager
    generation: GenerationService
    jobs: JobManager
    experiments: ExperimentService
    chess: ChessService
    started_at: float = field(default_factory=time.time)


def get_ctx(request: Request) -> ServerContext:
    return request.app.state.ctx

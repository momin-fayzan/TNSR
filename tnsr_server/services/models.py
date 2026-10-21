"""Model discovery, loading and safe shared access to the active inference engine."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

from tnsr_core.checkpoints import (
    CheckpointError,
    checkpoint_id,
    describe_checkpoint,
    discover_checkpoints,
    resolve_checkpoint_id,
)
from tnsr_core.inference import InferenceEngine


class ModelBusyError(RuntimeError):
    pass


class ModelManager:
    """Holds at most one loaded model. Loading and generation are serialized by locks:

    * ``_load_lock`` guards swapping the active engine;
    * each engine has a ``generate_lock`` so only one generation runs at a time
      (the model is CPU/GPU-bound; parallel decoding would only thrash).
    """

    def __init__(self, device: str = "auto"):
        self.device = device
        self._engine: InferenceEngine | None = None
        self._load_lock = threading.Lock()
        self.generate_lock = threading.Lock()
        self.loading_id: str | None = None
        self.load_error: str | None = None
        self.loaded_at: float | None = None

    @property
    def engine(self) -> InferenceEngine | None:
        return self._engine

    def current(self) -> dict[str, Any] | None:
        engine = self._engine
        if engine is None:
            return None
        return {**engine.describe(), "loaded_at": self.loaded_at}

    def status(self) -> dict[str, Any]:
        return {
            "loaded": self.current(),
            "loading": self.loading_id,
            "error": self.load_error,
        }

    def list_all(self) -> list[dict[str, Any]]:
        loaded = self._engine.model_id if self._engine else None
        models = discover_checkpoints()
        for m in models:
            m["loaded"] = m["id"] == loaded
        return models

    def describe(self, model_id: str) -> dict[str, Any]:
        path = resolve_checkpoint_id(model_id)
        info = describe_checkpoint(path)
        info["loaded"] = self._engine is not None and self._engine.model_id == info["id"]
        return info

    def load(self, model_id: str) -> dict[str, Any]:
        """Load (or return the already-loaded) model. Blocking: call from a worker thread."""
        path: Path = resolve_checkpoint_id(model_id)
        resolved_id = checkpoint_id(path)
        if self._engine is not None and self._engine.model_id == resolved_id:
            return self.current() or {}
        if not self._load_lock.acquire(timeout=60):
            raise ModelBusyError("Another model is still loading")
        try:
            self.loading_id = resolved_id
            self.load_error = None
            try:
                engine = InferenceEngine.from_checkpoint(path, self.device)
            except CheckpointError as exc:
                self.load_error = str(exc)
                raise
            # Swap under the generation lock so no stream sees a half-replaced engine.
            with self.generate_lock:
                self._engine = engine
                self.loaded_at = time.time()
            return self.current() or {}
        finally:
            self.loading_id = None
            self._load_lock.release()

    def unload(self) -> None:
        with self.generate_lock:
            self._engine = None
            self.loaded_at = None

    def ensure_loaded(self, model_id: str | None) -> InferenceEngine:
        if model_id:
            self.load(model_id)
        if self._engine is None:
            raise CheckpointError("No model is loaded. Select a model first.")
        return self._engine

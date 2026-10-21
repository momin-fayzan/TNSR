"""Bridges the blocking inference engine to async streaming with cancellation.

Each generation runs in a worker thread that pushes events onto an asyncio
queue. A ``threading.Event`` cancels it; the server sets the event when the
client disconnects or calls the cancel endpoint.
"""

from __future__ import annotations

import asyncio
import threading
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from tnsr_core.inference import GenerationConfig

from .models import ModelBusyError, ModelManager

QUEUE_WAIT_SECONDS = 120.0
CHAT_STOP = "\nUser:"


@dataclass
class ActiveGeneration:
    id: str
    cancel: threading.Event = field(default_factory=threading.Event)


class GenerationService:
    def __init__(self, models: ModelManager):
        self.models = models
        self.active: dict[str, ActiveGeneration] = {}

    def cancel(self, generation_id: str) -> bool:
        gen = self.active.get(generation_id)
        if gen is None:
            return False
        gen.cancel.set()
        return True

    def new_generation(self) -> ActiveGeneration:
        gen = ActiveGeneration(uuid.uuid4().hex[:16])
        self.active[gen.id] = gen
        return gen

    async def stream(
        self, gen: ActiveGeneration, prompt: str, cfg: GenerationConfig
    ) -> AsyncIterator[dict[str, Any]]:
        """Yield {'type': 'token'|'done'|'error', ...} dicts; always ends with done or error."""
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

        def put(item: dict[str, Any]) -> None:
            loop.call_soon_threadsafe(queue.put_nowait, item)

        def worker() -> None:
            acquired = self.models.generate_lock.acquire(timeout=QUEUE_WAIT_SECONDS)
            if not acquired:
                put(
                    {
                        "type": "error",
                        "detail": "The model is busy with another generation; try again.",
                    }
                )
                return
            try:
                if gen.cancel.is_set():
                    put(
                        {
                            "type": "done",
                            "stats": {"finish_reason": "cancelled", "generated_tokens": 0},
                        }
                    )
                    return
                engine = self.models.engine
                if engine is None:
                    put({"type": "error", "detail": "No model is loaded."})
                    return
                for event in engine.stream(prompt, cfg, cancel=gen.cancel):
                    if event.kind == "token":
                        put({"type": "token", "text": event.text})
                    else:
                        assert event.stats is not None
                        put(
                            {
                                "type": "done",
                                "stats": event.stats.to_dict(),
                                "model_id": engine.model_id,
                            }
                        )
            except ModelBusyError as exc:
                put({"type": "error", "detail": str(exc)})
            except Exception as exc:  # surface any engine failure to the client
                put({"type": "error", "detail": f"{type(exc).__name__}: {exc}"})
            finally:
                self.models.generate_lock.release()

        thread = threading.Thread(target=worker, name=f"gen-{gen.id}", daemon=True)
        thread.start()
        try:
            while True:
                item = await queue.get()
                yield item
                if item["type"] in {"done", "error"}:
                    break
        finally:
            # Client went away or the consumer stopped early: stop the worker.
            gen.cancel.set()
            self.active.pop(gen.id, None)


def build_chat_prompt(
    history: list[dict[str, Any]], user_text: str, tokenizer, budget_tokens: int
) -> str:
    """Format a conversation in the ``User:``/``TNSR:`` layout of the training corpus.

    Whole earlier turns are dropped (oldest first) until the prompt fits the
    token budget; the newest user turn is always kept.
    """
    tail = f"User: {user_text.strip()}\nTNSR:"
    turns: list[str] = []
    for msg in history:
        content = msg["content"].strip()
        if not content or msg.get("status") == "error":
            continue
        speaker = "User" if msg["role"] == "user" else "TNSR"
        turns.append(f"{speaker}: {content}\n")
    prompt = tail
    for turn in reversed(turns):
        candidate = turn + prompt
        if len(tokenizer.encode(candidate)) + 1 > budget_tokens:
            break
        prompt = candidate
    return prompt

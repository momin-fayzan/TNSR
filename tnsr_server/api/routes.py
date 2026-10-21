"""HTTP API. Streaming endpoints use Server-Sent Events over POST:

event: start   data: {"generation_id", "user_message", "assistant_message"}
event: token   data: {"text"}
event: done    data: {"message" | null, "stats"}
event: error   data: {"detail", "message" | null}
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import PlainTextResponse, StreamingResponse

from tnsr_core import __version__
from tnsr_core.checkpoints import CheckpointError, preferred_checkpoint
from tnsr_core.config import PRESETS_DIR, list_presets, load_config
from tnsr_core.data import DatasetError
from tnsr_core.device import select_device, system_summary
from tnsr_core.doctor import run_diagnostics
from tnsr_core.inference import GenerationConfigError
from tnsr_core.paths import checkpoints_dir, datasets_dir, experiments_dir, var_dir
from tnsr_server.context import ServerContext, get_ctx
from tnsr_server.services.generation import CHAT_STOP, build_chat_prompt
from tnsr_server.services.jobs import JobError
from tnsr_server.services.models import ModelBusyError

from .schemas import (
    AppSettings,
    CompareRequest,
    CompletionRequest,
    ConversationCreate,
    ConversationUpdate,
    GenerationSettings,
    JobCreate,
    LoadModelRequest,
    MessageCreate,
    RetryRequest,
)

router = APIRouter(prefix="/api")
DEFAULT_TITLE = "New conversation"


def sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


# -- system ----------------------------------------------------------------------------------


@router.get("/health")
def health(ctx: ServerContext = Depends(get_ctx)) -> dict[str, Any]:
    return {
        "status": "ok",
        "version": __version__,
        "uptime_seconds": round(time.time() - ctx.started_at, 1),
        "model": ctx.models.status(),
    }


@router.get("/system")
async def system(ctx: ServerContext = Depends(get_ctx)) -> dict[str, Any]:
    diagnostics = await run_in_threadpool(run_diagnostics, True)
    info = select_device(ctx.models.device)
    return {
        "version": __version__,
        "system": system_summary(),
        "device": {"selected": str(info.device), "reason": info.reason},
        "diagnostics": diagnostics,
        "storage": {
            "home": str(ctx.home),
            "experiments": str(experiments_dir()),
            "datasets": str(datasets_dir()),
            "checkpoints": str(checkpoints_dir()),
            "database": str(ctx.db.path),
            "database_schema_version": ctx.db.schema_version,
            "logs": str(var_dir() / "jobs"),
        },
        "model": ctx.models.status(),
    }


def _settings(ctx: ServerContext) -> AppSettings:
    raw = ctx.db.get_setting("app", None)
    try:
        return AppSettings.model_validate(raw) if raw else AppSettings()
    except ValueError:
        return AppSettings()


@router.get("/settings")
def get_settings(ctx: ServerContext = Depends(get_ctx)) -> AppSettings:
    return _settings(ctx)


@router.put("/settings")
def put_settings(body: AppSettings, ctx: ServerContext = Depends(get_ctx)) -> AppSettings:
    ctx.db.set_setting("app", body.model_dump())
    return body


# -- models ----------------------------------------------------------------------------------


@router.get("/models")
async def list_models(ctx: ServerContext = Depends(get_ctx)) -> dict[str, Any]:
    models = await run_in_threadpool(ctx.models.list_all)
    return {"models": models, **ctx.models.status()}


@router.get("/models/current")
def current_model(ctx: ServerContext = Depends(get_ctx)) -> dict[str, Any]:
    return ctx.models.status()


@router.post("/models/load")
async def load_model(
    body: LoadModelRequest, ctx: ServerContext = Depends(get_ctx)
) -> dict[str, Any]:
    try:
        loaded = await run_in_threadpool(ctx.models.load, body.model_id)
    except (CheckpointError, ValueError) as exc:
        raise HTTPException(404, str(exc)) from exc
    except ModelBusyError as exc:
        raise HTTPException(409, str(exc)) from exc
    return {"loaded": loaded}


@router.post("/models/unload")
async def unload_model(ctx: ServerContext = Depends(get_ctx)) -> dict[str, Any]:
    await run_in_threadpool(ctx.models.unload)
    return ctx.models.status()


@router.get("/models/{model_id:path}")
async def get_model(model_id: str, ctx: ServerContext = Depends(get_ctx)) -> dict[str, Any]:
    try:
        return await run_in_threadpool(ctx.models.describe, model_id)
    except (CheckpointError, ValueError) as exc:
        raise HTTPException(404, str(exc)) from exc


async def _ensure_model(ctx: ServerContext, model_id: str | None):
    """Load the requested (or default) model before streaming starts, so errors are plain JSON."""
    target = model_id
    if target is None and ctx.models.engine is None:
        target = _settings(ctx).default_model_id
        if target is None:
            models = await run_in_threadpool(ctx.models.list_all)
            choice = preferred_checkpoint(models)
            if choice is None:
                raise HTTPException(
                    409, "No trained models found. Train one first (tnsr train --config demo)."
                )
            target = choice["id"]
    try:
        return await run_in_threadpool(ctx.models.ensure_loaded, target)
    except (CheckpointError, ValueError) as exc:
        raise HTTPException(409, f"Model unavailable: {exc}") from exc
    except ModelBusyError as exc:
        raise HTTPException(409, str(exc)) from exc


# -- conversations ---------------------------------------------------------------------------


def _conversation_or_404(
    ctx: ServerContext, cid: str, with_messages: bool = False
) -> dict[str, Any]:
    conv = ctx.db.get_conversation(cid, with_messages=with_messages)
    if conv is None:
        raise HTTPException(404, "Conversation not found")
    return conv


@router.get("/conversations")
def list_conversations(ctx: ServerContext = Depends(get_ctx)) -> dict[str, Any]:
    return {"conversations": ctx.db.list_conversations()}


@router.post("/conversations", status_code=201)
def create_conversation(
    body: ConversationCreate, ctx: ServerContext = Depends(get_ctx)
) -> dict[str, Any]:
    settings = body.settings or _settings(ctx).generation
    return ctx.db.create_conversation(
        (body.title or DEFAULT_TITLE).strip() or DEFAULT_TITLE,
        body.model_id,
        body.mode,
        settings.model_dump(),
    )


@router.get("/conversations/{cid}")
def get_conversation(cid: str, ctx: ServerContext = Depends(get_ctx)) -> dict[str, Any]:
    return _conversation_or_404(ctx, cid, with_messages=True)


@router.patch("/conversations/{cid}")
def update_conversation(
    cid: str, body: ConversationUpdate, ctx: ServerContext = Depends(get_ctx)
) -> dict[str, Any]:
    _conversation_or_404(ctx, cid)
    conv = ctx.db.update_conversation(
        cid,
        title=body.title,
        model_id=body.model_id,
        mode=body.mode,
        settings=body.settings.model_dump() if body.settings else None,
    )
    assert conv is not None
    return conv


@router.delete("/conversations/{cid}", status_code=204)
def delete_conversation(cid: str, ctx: ServerContext = Depends(get_ctx)) -> None:
    if not ctx.db.delete_conversation(cid):
        raise HTTPException(404, "Conversation not found")


def _effective_settings(
    conv: dict[str, Any], override: GenerationSettings | None
) -> GenerationSettings:
    if override is not None:
        return override
    try:
        return GenerationSettings.model_validate(conv.get("settings") or {})
    except ValueError:
        return GenerationSettings()


async def _reply_stream(
    request: Request,
    ctx: ServerContext,
    conv: dict[str, Any],
    user_msg: dict[str, Any],
    history: list[dict[str, Any]],
    settings: GenerationSettings,
) -> AsyncIterator[str]:
    engine = ctx.models.engine
    assert engine is not None
    if conv["mode"] == "chat":
        budget = engine.block_size - min(settings.max_new_tokens, engine.block_size // 2)
        prompt = build_chat_prompt(history, user_msg["content"], engine.tokenizer, max(8, budget))
        cfg = settings.to_config(stop=[CHAT_STOP])
    else:
        prompt = user_msg["content"]
        cfg = settings.to_config()
    try:
        cfg.validate(engine.tokenizer.vocab_size)
    except GenerationConfigError as exc:
        yield sse("error", {"detail": str(exc), "message": None})
        return
    assistant = ctx.db.add_message(
        conv["id"],
        "assistant",
        "",
        status="streaming",
        model_id=engine.model_id,
        settings=settings.model_dump(),
    )
    gen = ctx.generation.new_generation()
    yield sse(
        "start", {"generation_id": gen.id, "user_message": user_msg, "assistant_message": assistant}
    )
    parts: list[str] = []
    status: str = "error"
    stats: dict[str, Any] | None = None
    error: str | None = "Generation did not finish"
    try:
        async for item in ctx.generation.stream(gen, prompt, cfg):
            if item["type"] == "token":
                text = item["text"]
                if not parts and conv["mode"] == "chat":
                    text = text.lstrip(" ")  # the format puts a space after "TNSR:"
                    if not text:
                        continue
                parts.append(text)
                yield sse("token", {"text": text})
            elif item["type"] == "done":
                stats = item["stats"]
                status = "cancelled" if stats.get("finish_reason") == "cancelled" else "complete"
                error = None
            else:
                error = item["detail"]
    finally:
        content = "".join(parts).rstrip() if conv["mode"] == "chat" else "".join(parts)
        if status == "error" and error == "Generation did not finish":
            status = "cancelled"  # client disconnected mid-stream
            error = None
        final = ctx.db.finish_message(assistant["id"], content, status, stats, error)
    if status == "error":
        yield sse("error", {"detail": error, "message": final})
    else:
        yield sse("done", {"message": final, "stats": stats})


@router.post("/conversations/{cid}/messages")
async def send_message(
    cid: str, body: MessageCreate, request: Request, ctx: ServerContext = Depends(get_ctx)
):
    conv = _conversation_or_404(ctx, cid, with_messages=True)
    model_id = body.model_id or conv["model_id"]
    await _ensure_model(ctx, model_id)
    engine = ctx.models.engine
    assert engine is not None
    if conv["model_id"] != engine.model_id:
        ctx.db.update_conversation(cid, model_id=engine.model_id)
    if conv["title"] == DEFAULT_TITLE and not conv["messages"]:
        title = " ".join(body.content.split())[:60] or DEFAULT_TITLE
        ctx.db.update_conversation(cid, title=title)
    settings = _effective_settings(conv, body.settings)
    history = conv["messages"]
    user_msg = ctx.db.add_message(cid, "user", body.content)
    return StreamingResponse(
        _reply_stream(request, ctx, conv, user_msg, history, settings),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )


@router.post("/conversations/{cid}/messages/{mid}/retry")
async def retry_message(
    cid: str,
    mid: str,
    request: Request,
    body: RetryRequest | None = None,
    ctx: ServerContext = Depends(get_ctx),
):
    conv = _conversation_or_404(ctx, cid, with_messages=True)
    messages = conv["messages"]
    target = next((m for m in messages if m["id"] == mid), None)
    if target is None or target["role"] != "assistant":
        raise HTTPException(404, "Assistant message not found in this conversation")
    if target["status"] == "streaming":
        raise HTTPException(409, "That message is still generating")
    user_msg = next(
        (m for m in reversed(messages) if m["seq"] < target["seq"] and m["role"] == "user"), None
    )
    if user_msg is None:
        raise HTTPException(409, "No user message to respond to")
    await _ensure_model(ctx, conv["model_id"])
    ctx.db.delete_messages_from(cid, target["seq"])
    history = [m for m in messages if m["seq"] < user_msg["seq"]]
    settings = _effective_settings(conv, body.settings if body else None)
    return StreamingResponse(
        _reply_stream(request, ctx, conv, user_msg, history, settings),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )


@router.post("/generations/{generation_id}/cancel")
def cancel_generation(generation_id: str, ctx: ServerContext = Depends(get_ctx)) -> dict[str, Any]:
    return {"cancelled": ctx.generation.cancel(generation_id)}


@router.post("/generate")
async def generate(body: CompletionRequest, ctx: ServerContext = Depends(get_ctx)):
    """Raw text completion (no conversation persistence)."""
    await _ensure_model(ctx, body.model_id)
    engine = ctx.models.engine
    assert engine is not None
    cfg = body.settings.to_config(stop=body.stop)
    try:
        cfg.validate(engine.tokenizer.vocab_size)
    except GenerationConfigError as exc:
        raise HTTPException(422, str(exc)) from exc
    gen = ctx.generation.new_generation()

    async def events() -> AsyncIterator[str]:
        yield sse("start", {"generation_id": gen.id, "model_id": engine.model_id})
        async for item in ctx.generation.stream(gen, body.prompt, cfg):
            if item["type"] == "token":
                yield sse("token", {"text": item["text"]})
            elif item["type"] == "done":
                yield sse("done", {"stats": item["stats"], "message": None})
            else:
                yield sse("error", {"detail": item["detail"], "message": None})

    return StreamingResponse(events(), media_type="text/event-stream", headers=SSE_HEADERS)


# -- lab: experiments, datasets, evaluations, jobs ---------------------------------------------


@router.get("/experiments")
async def list_experiments(ctx: ServerContext = Depends(get_ctx)) -> dict[str, Any]:
    return {"experiments": await run_in_threadpool(ctx.experiments.list_all)}


@router.post("/experiments/compare")
async def compare_experiments(
    body: CompareRequest, ctx: ServerContext = Depends(get_ctx)
) -> dict[str, Any]:
    try:
        return await run_in_threadpool(ctx.experiments.comparison, body.run_ids)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(404, str(exc)) from exc


@router.get("/experiments/{run_id}")
async def get_experiment(run_id: str, ctx: ServerContext = Depends(get_ctx)) -> dict[str, Any]:
    try:
        return await run_in_threadpool(ctx.experiments.detail, run_id)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(404, str(exc)) from exc


@router.post("/experiments/{run_id}/stop")
def stop_experiment(run_id: str) -> dict[str, Any]:
    """Gracefully stop a running training process (also ones started from the CLI)."""
    from tnsr_server.services.jobs import stop_run_process

    try:
        return {"signalled": stop_run_process(run_id)}
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(404, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc


@router.get("/experiments/{run_id}/export")
async def export_experiment(
    run_id: str, format: str = "json", ctx: ServerContext = Depends(get_ctx)
):
    try:
        if format == "csv":
            body = await run_in_threadpool(ctx.experiments.export_metrics_csv, run_id)
            return PlainTextResponse(
                body,
                media_type="text/csv",
                headers={"Content-Disposition": f'attachment; filename="{run_id}-metrics.csv"'},
            )
        if format != "json":
            raise HTTPException(422, "format must be json or csv")
        data = await run_in_threadpool(ctx.experiments.export_json, run_id)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(404, str(exc)) from exc
    return PlainTextResponse(
        json.dumps(data, indent=2),
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{run_id}.json"'},
    )


@router.get("/evaluations")
async def list_evaluations(
    kind: str | None = None, experiment_id: str | None = None, ctx: ServerContext = Depends(get_ctx)
) -> dict[str, Any]:
    await run_in_threadpool(ctx.experiments.sync)
    return {"evaluations": ctx.db.list_evaluations(experiment_id, kind)}


@router.get("/datasets")
async def datasets() -> dict[str, Any]:
    from tnsr_server.services.experiments import list_datasets

    return {"datasets": await run_in_threadpool(list_datasets)}


@router.get("/presets")
def presets() -> dict[str, Any]:
    out = []
    for name in list_presets():
        try:
            cfg = load_config(name)
        except (ValueError, DatasetError):
            continue
        out.append(
            {
                "name": name,
                "config": cfg.to_dict(),
                "toml": (PRESETS_DIR / f"{name}.toml").read_text(),
            }
        )
    return {"presets": out}


@router.get("/jobs")
def list_jobs(ctx: ServerContext = Depends(get_ctx)) -> dict[str, Any]:
    return {"jobs": ctx.jobs.list_all()}


@router.post("/jobs", status_code=201)
def create_job(body: JobCreate, ctx: ServerContext = Depends(get_ctx)) -> dict[str, Any]:
    try:
        return ctx.jobs.start(body.kind, body.params)
    except JobError as exc:
        raise HTTPException(422, str(exc)) from exc


@router.get("/jobs/{job_id}")
def get_job(job_id: str, ctx: ServerContext = Depends(get_ctx)) -> dict[str, Any]:
    job = ctx.jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    job = dict(job)
    job["log"] = ctx.jobs.log_tail(job)
    job.pop("log_path", None)
    return job


@router.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str, ctx: ServerContext = Depends(get_ctx)) -> dict[str, Any]:
    try:
        return ctx.jobs.cancel(job_id)
    except KeyError as exc:
        raise HTTPException(404, "Job not found") from exc

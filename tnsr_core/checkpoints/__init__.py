"""Checkpoint format, atomic saving, validated loading and discovery.

A checkpoint is a directory:

    model.pt          model weights only (loaded with ``weights_only=True``)
    meta.json         format version, model config, tokenizer identity, step,
                      dataset identity, training settings, metrics
    tokenizer.json    the exact tokenizer used, so inference is self-contained
    train_state.pt    optimizer / RNG / sampler state (optional; training only)

Inference needs only the first three files. Checkpoints live under a training
run (``experiments/<run>/checkpoints/step_XXXXXXX``) or as standalone imports
(``checkpoints/<name>``).
"""

from __future__ import annotations

import json
import shutil
import tarfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from tnsr_core.config import ModelConfig, model_config_from_dict
from tnsr_core.model import TNSRModel
from tnsr_core.paths import checkpoints_dir, experiments_dir, safe_join, tnsr_home, validate_name
from tnsr_core.tokenizer import Tokenizer, load_tokenizer, save_tokenizer

CHECKPOINT_FORMAT_VERSION = 1
REQUIRED_FILES = ("model.pt", "meta.json", "tokenizer.json")


class CheckpointError(RuntimeError):
    """A checkpoint is missing, corrupt or incompatible."""


def save_checkpoint(
    path: Path,
    model: TNSRModel,
    tokenizer: Tokenizer,
    meta: dict[str, Any],
    train_state: dict[str, Any] | None = None,
) -> Path:
    """Atomically write a checkpoint directory (write to ``.tmp`` then rename)."""
    path = Path(path)
    if path.exists():
        raise CheckpointError(f"Refusing to overwrite existing checkpoint {path}")
    tmp = path.with_name(path.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save(state, tmp / "model.pt")
    save_tokenizer(tokenizer, tmp / "tokenizer.json")
    full_meta = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "model_config": model.config.__dict__,
        "num_parameters": model.num_parameters(),
        "tokenizer": tokenizer.describe(),
        **meta,
    }
    (tmp / "meta.json").write_text(json.dumps(full_meta, indent=2))
    if train_state is not None:
        torch.save(train_state, tmp / "train_state.pt")
    tmp.rename(path)
    return path


def read_meta(path: Path) -> dict[str, Any]:
    meta_path = Path(path) / "meta.json"
    if not meta_path.is_file():
        raise CheckpointError(f"No checkpoint metadata at {meta_path}")
    try:
        meta = json.loads(meta_path.read_text())
    except json.JSONDecodeError as exc:
        raise CheckpointError(f"Corrupt checkpoint metadata in {meta_path}: {exc}") from exc
    if meta.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise CheckpointError(
            f"Checkpoint {path} has format version {meta.get('format_version')!r}; "
            f"this TNSR build reads version {CHECKPOINT_FORMAT_VERSION}"
        )
    return meta


@dataclass
class LoadedCheckpoint:
    path: Path
    model: TNSRModel
    tokenizer: Tokenizer
    meta: dict[str, Any]
    config: ModelConfig = field(init=False)

    def __post_init__(self) -> None:
        self.config = self.model.config


def load_checkpoint(path: Path, device: torch.device | str = "cpu") -> LoadedCheckpoint:
    """Load weights + tokenizer for inference, validating every compatibility invariant."""
    path = Path(path)
    if not path.is_dir():
        raise CheckpointError(f"Checkpoint not found: {path}")
    missing = [f for f in REQUIRED_FILES if not (path / f).is_file()]
    if missing:
        raise CheckpointError(f"Checkpoint {path} is incomplete; missing {', '.join(missing)}")
    meta = read_meta(path)
    try:
        cfg = model_config_from_dict(meta["model_config"])
    except (KeyError, ValueError) as exc:
        raise CheckpointError(f"Invalid model config in {path}: {exc}") from exc
    try:
        tokenizer = load_tokenizer(path / "tokenizer.json")
    except (ValueError, KeyError) as exc:
        raise CheckpointError(f"Invalid tokenizer in {path}: {exc}") from exc
    expected_tok = meta.get("tokenizer", {}).get("identity")
    if expected_tok and tokenizer.identity != expected_tok:
        raise CheckpointError(
            f"Tokenizer mismatch in {path}: file is {tokenizer.identity}, metadata says {expected_tok}"
        )
    if tokenizer.vocab_size != cfg.vocab_size:
        raise CheckpointError(
            f"Vocabulary size mismatch in {path}: tokenizer has {tokenizer.vocab_size} tokens, "
            f"model expects {cfg.vocab_size}"
        )
    try:
        state = torch.load(path / "model.pt", map_location="cpu", weights_only=True)
    except Exception as exc:  # torch raises a variety of unpickling errors
        raise CheckpointError(f"Could not read weights from {path / 'model.pt'}: {exc}") from exc
    model = TNSRModel(cfg)
    result = model.load_state_dict(state, strict=False)
    if result.missing_keys or result.unexpected_keys:
        raise CheckpointError(
            f"Weights in {path} do not match the model config "
            f"(missing={result.missing_keys[:5]}, unexpected={result.unexpected_keys[:5]})"
        )
    model.to(device)
    model.eval()
    return LoadedCheckpoint(path, model, tokenizer, meta)


def checkpoint_id(path: Path) -> str:
    """Stable id: path relative to TNSR_HOME using forward slashes."""
    path = Path(path).resolve()
    try:
        return path.relative_to(tnsr_home()).as_posix()
    except ValueError:
        return path.as_posix()


def resolve_checkpoint(ref: str) -> Path:
    """Resolve a checkpoint id, a run id (-> its best checkpoint) or a directory path.

    Ids are resolved inside TNSR_HOME only; arbitrary absolute paths are allowed
    for CLI use but the API layer passes ``ref`` through :func:`resolve_checkpoint_id`.
    """
    path = Path(ref)
    if path.is_dir() and (path / "meta.json").is_file():
        return path.resolve()
    if path.is_dir() and (path / "run.json").is_file():
        return best_checkpoint_of_run(path)
    return resolve_checkpoint_id(ref)


def resolve_checkpoint_id(ref: str) -> Path:
    """Resolve an id strictly within TNSR_HOME (used for untrusted API input)."""
    parts = [p for p in ref.split("/") if p]
    if not parts:
        raise CheckpointError("Empty checkpoint id")
    for p in parts:
        validate_name(p, "checkpoint path component")
    home = tnsr_home()
    candidate = safe_join(home, *parts)
    if (candidate / "meta.json").is_file():
        return candidate
    if (candidate / "run.json").is_file():
        return best_checkpoint_of_run(candidate)
    if len(parts) == 1:
        run = safe_join(experiments_dir(), parts[0])
        if (run / "run.json").is_file():
            return best_checkpoint_of_run(run)
        standalone = safe_join(checkpoints_dir(), parts[0])
        if (standalone / "meta.json").is_file():
            return standalone
    raise CheckpointError(f"Checkpoint {ref!r} not found (try: tnsr models list)")


def run_checkpoint_index(run_dir: Path) -> dict[str, Any]:
    index_path = run_dir / "checkpoints.json"
    if not index_path.is_file():
        return {"checkpoints": [], "best": None, "latest": None}
    return json.loads(index_path.read_text())


def best_checkpoint_of_run(run_dir: Path) -> Path:
    index = run_checkpoint_index(run_dir)
    for key in ("best", "latest"):
        name = index.get(key)
        if name and (run_dir / "checkpoints" / name / "meta.json").is_file():
            return run_dir / "checkpoints" / name
    existing = sorted((run_dir / "checkpoints").glob("step_*"))
    existing = [p for p in existing if (p / "meta.json").is_file()]
    if not existing:
        raise CheckpointError(f"Run {run_dir.name} has no checkpoints yet")
    return existing[-1]


def results_dir_for(ckpt_path: Path) -> Path:
    """Where evaluations/benchmarks for a checkpoint are stored (survives checkpoint rotation)."""
    ckpt_path = Path(ckpt_path)
    if ckpt_path.parent.name == "checkpoints" and (ckpt_path.parent.parent / "run.json").is_file():
        return ckpt_path.parent.parent
    return ckpt_path


def _latest_results(root: Path, kind: str, ckpt_name: str | None) -> dict[str, Any] | None:
    folder = root / kind
    if not folder.is_dir():
        return None
    files = sorted(folder.glob("*.json"))
    for f in reversed(files):
        try:
            data = json.loads(f.read_text())
        except json.JSONDecodeError:
            continue
        if ckpt_name is None or data.get("checkpoint_name") == ckpt_name:
            return data
    return None


def describe_checkpoint(path: Path) -> dict[str, Any]:
    """Metadata summary for listings (reads JSON only; never loads weights)."""
    meta = read_meta(path)
    run_dir = results_dir_for(path)
    is_run = run_dir != path
    index = run_checkpoint_index(run_dir) if is_run else {}
    files_ok = all((path / f).is_file() for f in REQUIRED_FILES)
    size = sum(f.stat().st_size for f in path.iterdir() if f.is_file())
    return {
        "id": checkpoint_id(path),
        "name": path.name,
        "run_id": run_dir.name if is_run else None,
        "run_name": meta.get("run_name"),
        "step": meta.get("step"),
        "num_parameters": meta.get("num_parameters"),
        "model_config": meta.get("model_config"),
        "tokenizer": meta.get("tokenizer"),
        "dataset": meta.get("dataset"),
        "metrics": meta.get("metrics", {}),
        "saved_at": meta.get("saved_at"),
        "is_best": bool(is_run and index.get("best") == path.name),
        "is_latest": bool(is_run and index.get("latest") == path.name),
        "has_train_state": (path / "train_state.pt").is_file(),
        "complete": files_ok,
        "size_bytes": size,
        "evaluation": _latest_results(run_dir, "evals", path.name),
        "benchmark": _latest_results(run_dir, "benchmarks", path.name),
    }


def discover_checkpoints() -> list[dict[str, Any]]:
    """All valid checkpoints under experiments/ and checkpoints/, newest first."""
    found: list[dict[str, Any]] = []
    candidates: list[Path] = []
    if experiments_dir().is_dir():
        candidates += [p for p in experiments_dir().glob("*/checkpoints/*") if p.is_dir()]
    if checkpoints_dir().is_dir():
        candidates += [p for p in checkpoints_dir().iterdir() if p.is_dir()]
    for p in candidates:
        if p.name.endswith((".tmp", ".partial")):
            continue
        try:
            found.append(describe_checkpoint(p))
        except (CheckpointError, OSError, json.JSONDecodeError):
            continue
    found.sort(key=lambda d: (d.get("saved_at") or ""), reverse=True)
    return found


def export_checkpoint(path: Path, out_file: Path, include_train_state: bool = False) -> Path:
    """Package a checkpoint as a portable .tar.gz (weights, tokenizer, metadata)."""
    path = Path(path)
    read_meta(path)
    out_file = Path(out_file)
    if out_file.exists():
        raise CheckpointError(f"Refusing to overwrite {out_file}")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out_file, "w:gz") as tar:
        for name in REQUIRED_FILES + (("train_state.pt",) if include_train_state else ()):
            if (path / name).is_file():
                tar.add(path / name, arcname=f"checkpoint/{name}")
    return out_file


def import_checkpoint(archive: Path, name: str) -> Path:
    """Unpack an exported archive into ``checkpoints/<name>`` after validating it."""
    validate_name(name, "checkpoint name")
    dest = safe_join(checkpoints_dir(), name)
    if dest.exists():
        raise CheckpointError(f"Checkpoint {name!r} already exists at {dest}")
    allowed = {f"checkpoint/{n}" for n in (*REQUIRED_FILES, "train_state.pt")}
    tmp = dest.with_name(name + ".partial")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    try:
        with tarfile.open(archive, "r:gz") as tar:
            for member in tar.getmembers():
                if member.name not in allowed or not member.isfile():
                    raise CheckpointError(f"Unexpected entry in archive: {member.name}")
                src = tar.extractfile(member)
                assert src is not None
                (tmp / Path(member.name).name).write_bytes(src.read())
        load_checkpoint(tmp)  # full validation before the import becomes visible
        tmp.rename(dest)
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return dest

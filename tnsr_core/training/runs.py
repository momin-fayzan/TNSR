"""Experiment run directories: metadata, live status and metric logs."""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

from tnsr_core import __version__
from tnsr_core.device import system_summary
from tnsr_core.paths import experiments_dir, safe_join, validate_name


def write_json_atomic(path: Path, data: Any) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, path)


def git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=Path(__file__).resolve().parent,
        )
        return out.stdout.strip() or None if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def environment_info(device: str, device_reason: str) -> dict[str, Any]:
    return {
        **system_summary(),
        "tnsr_version": __version__,
        "git_commit": git_commit(),
        "device": device,
        "device_reason": device_reason,
    }


def new_run_dir(name: str, root: Path | None = None) -> Path:
    """Create a fresh, uniquely named run directory; never reuses an existing one."""
    validate_name(name, "run name")
    root = root or experiments_dir()
    root.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    for suffix in [""] + [f"-{i}" for i in range(1, 100)]:
        run_dir = safe_join(root, f"{name}-{stamp}{suffix}")
        try:
            run_dir.mkdir()
            (run_dir / "checkpoints").mkdir()
            return run_dir
        except FileExistsError:
            continue
    raise RuntimeError("Could not allocate a unique run directory")


def append_metrics(run_dir: Path, record: dict[str, Any]) -> None:
    with (run_dir / "metrics.jsonl").open("a") as fh:
        fh.write(json.dumps(record) + "\n")
        fh.flush()


def read_metrics(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "metrics.jsonl"
    if not path.is_file():
        return []
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # tolerate a partially written last line during training
    return out


def truncate_metrics_after(run_dir: Path, step: int) -> None:
    """On resume, drop records logged after the checkpoint we resume from."""
    path = run_dir / "metrics.jsonl"
    if not path.is_file():
        return
    keep = [r for r in read_metrics(run_dir) if r.get("step", 0) <= step]
    tmp = path.with_suffix(".jsonl.tmp")
    tmp.write_text("".join(json.dumps(r) + "\n" for r in keep))
    os.replace(tmp, path)


def read_run(run_dir: Path) -> dict[str, Any]:
    run = json.loads((run_dir / "run.json").read_text())
    status_path = run_dir / "status.json"
    status: dict[str, Any] = {}
    if status_path.is_file():
        try:
            status = json.loads(status_path.read_text())
        except json.JSONDecodeError:
            status = {}
    run["status"] = status
    return run


def list_runs(root: Path | None = None) -> list[dict[str, Any]]:
    root = root or experiments_dir()
    if not root.is_dir():
        return []
    runs = []
    for d in root.iterdir():
        if (d / "run.json").is_file():
            try:
                runs.append(read_run(d))
            except (OSError, json.JSONDecodeError):
                continue
    runs.sort(key=lambda r: r.get("created_at", ""), reverse=True)
    return runs


def resolve_run(ref: str) -> Path:
    path = Path(ref)
    if path.is_dir() and (path / "run.json").is_file():
        return path.resolve()
    validate_name(ref, "run id")
    run_dir = safe_join(experiments_dir(), ref)
    if not (run_dir / "run.json").is_file():
        raise FileNotFoundError(f"Run {ref!r} not found in {experiments_dir()}")
    return run_dir

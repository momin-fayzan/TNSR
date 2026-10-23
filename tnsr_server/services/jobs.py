"""Long-running work (training, evaluation, benchmarks, dataset preparation) as subprocesses.

Each job runs ``python -m tnsr_core.cli ...`` in its own process with output
captured to ``var/jobs/<id>.log``. Training therefore never blocks request
handling, survives as an independent process, and is stopped gracefully with
SIGINT (the trainer saves a checkpoint before exiting).
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import psutil

from tnsr_core.config import ConfigError, list_presets, load_config, parse_override
from tnsr_core.paths import validate_name, var_dir
from tnsr_core.training import read_run, resolve_run
from tnsr_server.persistence import Database

JOB_KINDS = {"train", "resume", "evaluate", "benchmark", "prepare", "chess_prepare"}
ACTIVE_STATES = {"pending", "running", "stopping"}


class JobError(ValueError):
    pass


def _flag(name: str, value: Any) -> list[str]:
    return [f"--{name}", str(value)]


class JobManager:
    def __init__(self, db: Database, home: Path):
        self.db = db
        self.home = home
        self.procs: dict[str, subprocess.Popen] = {}
        self._lock = threading.Lock()
        self.log_dir = var_dir() / "jobs"
        self.log_dir.mkdir(parents=True, exist_ok=True)

    # -- command construction ------------------------------------------------------------

    def build_command(self, kind: str, params: dict[str, Any], job_id: str) -> list[str]:
        base = [sys.executable, "-m", "tnsr_core.cli"]
        if kind == "train":
            preset = params.get("config", "demo")
            if preset not in list_presets():
                raise JobError(f"Unknown preset {preset!r}; available: {', '.join(list_presets())}")
            overrides = [str(o) for o in params.get("overrides", [])]
            name = params.get("name")
            try:
                parsed = dict(parse_override(o) for o in overrides)
                if name:
                    validate_name(name, "run name")
                    parsed["name"] = name
                load_config(preset, parsed)  # validate before spawning
            except (ConfigError, ValueError) as exc:
                raise JobError(str(exc)) from exc
            cmd = [
                *base,
                "train",
                "--config",
                preset,
                "--run-id-file",
                str(self.log_dir / f"{job_id}.run"),
            ]
            if name:
                cmd += _flag("name", name)
            for o in overrides:
                cmd += ["--set", o]
            return cmd
        if kind == "resume":
            run_id = params.get("run_id", "")
            try:
                resolve_run(run_id)
            except (FileNotFoundError, ValueError) as exc:
                raise JobError(str(exc)) from exc
            cmd = [*base, "train", "--resume", run_id]
            if params.get("max_steps"):
                cmd += _flag("max-steps", int(params["max_steps"]))
            return cmd
        if kind in {"evaluate", "benchmark"}:
            model_id = params.get("model_id")
            if not model_id:
                raise JobError("model_id is required")
            from tnsr_core.checkpoints import CheckpointError, resolve_checkpoint_id

            try:
                path = resolve_checkpoint_id(model_id)
            except (CheckpointError, ValueError) as exc:
                raise JobError(str(exc)) from exc
            from tnsr_core.checkpoints import read_meta

            if read_meta(path).get("task") == "chess":
                if kind == "benchmark":
                    raise JobError("Benchmarks apply to text models; evaluate chess models instead")
                games = int(params.get("games", 10))
                if not 0 <= games <= 200:
                    raise JobError("games must be in [0, 200]")
                return [*base, "chess", "evaluate", str(path), "--json", "--puzzles",
                        str(int(params.get("puzzles", 1000))), "--games", str(games)]  # fmt: skip
            cmd = [*base, kind, str(path), "--json"]
            if kind == "evaluate" and params.get("max_tokens"):
                cmd += _flag("max-tokens", int(params["max_tokens"]))
            return cmd
        if kind == "chess_prepare":
            name = params.get("name", "")
            month = str(params.get("month", "2013-01"))
            try:
                validate_name(name, "dataset name")
            except ValueError as exc:
                raise JobError(str(exc)) from exc
            if not re.fullmatch(r"20\d\d-(0[1-9]|1[0-2])", month):
                raise JobError("month must look like 2013-01")
            positions = int(params.get("max_positions", 20000))
            nodes = int(params.get("nodes", 2000))
            if not 100 <= positions <= 50_000_000 or not 50 <= nodes <= 1_000_000:
                raise JobError("max_positions must be in [100, 50M] and nodes in [50, 1M]")
            return [*base, "chess", "prepare", "--month", month, "--name", name,
                    "--max-positions", str(positions), "--nodes", str(nodes)]  # fmt: skip
        if kind == "prepare":
            name = params.get("name", "")
            try:
                validate_name(name, "dataset name")
            except ValueError as exc:
                raise JobError(str(exc)) from exc
            source = params.get("source", "synthetic")
            if source != "synthetic":
                raise JobError(
                    "Only the synthetic source can be prepared from the UI; use the CLI for files"
                )
            tokenizer = params.get("tokenizer", "bpe")
            if tokenizer not in {"bpe", "char"}:
                raise JobError("tokenizer must be bpe or char")
            vocab = int(params.get("vocab_size", 2048))
            docs = int(params.get("docs", 20000))
            if not 257 <= vocab <= 65536 or not 10 <= docs <= 500_000:
                raise JobError("vocab_size must be in [257, 65536] and docs in [10, 500000]")
            return [
                *base, "data", "prepare", "--name", name, "--tokenizer", tokenizer,
                "--vocab-size", str(vocab), "--docs", str(docs), "--seed", str(int(params.get("seed", 0))),
            ]  # fmt: skip
        raise JobError(f"Unknown job kind {kind!r}; expected one of {sorted(JOB_KINDS)}")

    # -- lifecycle --------------------------------------------------------------------

    def start(self, kind: str, params: dict[str, Any]) -> dict[str, Any]:
        if kind not in JOB_KINDS:
            raise JobError(f"Unknown job kind {kind!r}")
        if kind in {"train", "resume"} and any(
            j["kind"] in {"train", "resume"} and j["status"] in ACTIVE_STATES
            for j in self.list_all()
        ):
            raise JobError(
                "A training job is already running; stop it first or wait for it to finish"
            )
        if kind == "resume":
            for j in self.list_all():
                if j["run_id"] == params.get("run_id") and j["status"] in ACTIVE_STATES:
                    raise JobError("That run is already training")
        job = self.db.create_job(kind, params)
        try:
            cmd = self.build_command(kind, params, job["id"])
        except JobError as exc:
            self.db.update_job(job["id"], status="failed", error=str(exc), finished_at=time.time())
            raise
        log_path = self.log_dir / f"{job['id']}.log"
        env = {**os.environ, "TNSR_HOME": str(self.home), "PYTHONUNBUFFERED": "1", "COLUMNS": "120"}
        with log_path.open("w") as log:
            log.write("$ tnsr " + " ".join(cmd[3:]) + "\n")
            log.flush()
            proc = subprocess.Popen(
                cmd, stdout=log, stderr=subprocess.STDOUT, cwd=str(self.home), env=env,
                start_new_session=True,
            )  # fmt: skip
        with self._lock:
            self.procs[job["id"]] = proc
        self.db.update_job(
            job["id"], status="running", pid=proc.pid, log_path=str(log_path), started_at=time.time(),
            run_id=params.get("run_id") if kind == "resume" else None,
        )  # fmt: skip
        return self.get(job["id"]) or job

    def cancel(self, job_id: str) -> dict[str, Any]:
        job = self.db.get_job(job_id)
        if job is None:
            raise KeyError(job_id)
        if job["status"] not in ACTIVE_STATES:
            return job
        pid = job["pid"]
        if pid and psutil.pid_exists(pid):
            with contextlib.suppress(ProcessLookupError):
                os.kill(pid, signal.SIGINT)  # trainer saves a checkpoint and exits cleanly
        self.db.update_job(job_id, status="stopping")
        return self.get(job_id) or job

    def refresh(self, job: dict[str, Any]) -> dict[str, Any]:
        """Reconcile a job row with its process; collect results when it finishes."""
        if job["status"] not in ACTIVE_STATES:
            return job
        jid = job["id"]
        run_file = self.log_dir / f"{jid}.run"
        if not job["run_id"] and run_file.is_file():
            job["run_id"] = run_file.read_text().strip()
            self.db.update_job(jid, run_id=job["run_id"])
        with self._lock:
            proc = self.procs.get(jid)
        code: int | None
        if proc is not None:
            code = proc.poll()
            if code is None:
                return job
        else:
            # Process from a previous server session: alive if the pid still exists.
            pid = job["pid"]
            if pid and psutil.pid_exists(pid):
                try:
                    if psutil.Process(pid).status() != psutil.STATUS_ZOMBIE:
                        return job
                except psutil.NoSuchProcess:
                    pass
            code = None
        status, error, result = self._outcome(job, code)
        self.db.update_job(
            jid, status=status, exit_code=code, error=error, result=result, finished_at=time.time()
        )
        with self._lock:
            self.procs.pop(jid, None)
        return self.db.get_job(jid) or job

    def _outcome(self, job: dict[str, Any], code: int | None) -> tuple[str, str | None, Any]:
        log_tail = self.log_tail(job, 200_000)
        if job["kind"] in {"train", "resume"} and job.get("run_id"):
            try:
                state = read_run(resolve_run(job["run_id"])).get("status", {}).get("state")
            except (FileNotFoundError, ValueError, json.JSONDecodeError):
                state = None
            if state in {"completed", "interrupted"}:
                return (
                    ("completed" if state == "completed" else "stopped"),
                    None,
                    {"run_state": state},
                )
            if state == "failed":
                return "failed", _last_error(log_tail), {"run_state": state}
        if code == 0:
            result = None
            if job["kind"] in {"evaluate", "benchmark"}:
                result = _parse_json_tail(log_tail)
            return "completed", None, result
        if code is None:
            return "lost", "Process ended while the server was not watching it", None
        if code in (130, -signal.SIGINT) or job["status"] == "stopping":
            return "stopped", None, None
        return "failed", _last_error(log_tail) or f"exit code {code}", None

    def get(self, job_id: str) -> dict[str, Any] | None:
        job = self.db.get_job(job_id)
        return self.refresh(job) if job else None

    def list_all(self) -> list[dict[str, Any]]:
        return [self.refresh(j) for j in self.db.list_jobs()]

    def log_tail(self, job: dict[str, Any], max_chars: int = 20000) -> str:
        path = job.get("log_path")
        if not path or not Path(path).is_file():
            return ""
        with open(path, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            fh.seek(max(0, size - max_chars))
            return fh.read().decode("utf-8", errors="replace")

    def shutdown(self) -> None:
        """Leave training processes running (they are independent) but stop watching them."""
        with self._lock:
            self.procs.clear()


def stop_run_process(run_id: str) -> bool:
    """Send SIGINT to the training process recorded in a run's status.json.

    The pid is only signalled if it is alive and is a TNSR training command, so
    a stale status file can never cause an unrelated process to be interrupted.
    """
    run = read_run(resolve_run(run_id))
    status = run.get("status") or {}
    pid = status.get("pid")
    if status.get("state") != "running" or not pid:
        raise RuntimeError("Run is not running")
    try:
        proc = psutil.Process(int(pid))
        cmdline = " ".join(proc.cmdline())
    except (psutil.NoSuchProcess, psutil.AccessDenied) as exc:
        raise RuntimeError("Training process is no longer alive") from exc
    if "tnsr" not in cmdline or "train" not in cmdline:
        raise RuntimeError("Recorded pid does not belong to a TNSR training process")
    proc.send_signal(signal.SIGINT)
    return True


def _last_error(log: str) -> str | None:
    lines = [ln.strip() for ln in log.splitlines() if ln.strip()]
    for ln in reversed(lines):
        if ln.startswith("error:") or "Error" in ln:
            return ln[:500]
    return lines[-1][:500] if lines else None


def _parse_json_tail(log: str) -> Any:
    start = log.find("\n{")
    if start < 0:
        return None
    try:
        return json.loads(log[start + 1 :])
    except json.JSONDecodeError:
        return None

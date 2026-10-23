"""Training engine: AdamW, LR schedule, clipping, evaluation, checkpointing, resume."""

from __future__ import annotations

import contextlib
import math
import random
import shutil
import signal
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import torch

from tnsr_core.checkpoints import CheckpointError, run_checkpoint_index, save_checkpoint
from tnsr_core.config import RunConfig
from tnsr_core.device import autocast_dtype, select_device
from tnsr_core.model import TNSRModel

from .runs import (
    append_metrics,
    environment_info,
    new_run_dir,
    read_run,
    truncate_metrics_after,
    write_json_atomic,
)
from .tasks import make_task


class TrainingError(RuntimeError):
    pass


def lr_at(step: int, cfg) -> float:
    """Learning rate for optimizer step ``step`` (0-based): linear warmup then decay to min_lr."""
    if cfg.warmup_steps and step < cfg.warmup_steps:
        return cfg.learning_rate * (step + 1) / cfg.warmup_steps
    if cfg.lr_schedule == "constant":
        return cfg.learning_rate
    span = max(1, cfg.max_steps - cfg.warmup_steps)
    progress = min(1.0, (step - cfg.warmup_steps) / span)
    if cfg.lr_schedule == "linear":
        coeff = 1.0 - progress
    else:
        coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)


@dataclass
class TrainResult:
    run_dir: Path
    status: str
    step: int
    best_val_loss: float | None
    last_checkpoint: Path | None


class _StopFlag:
    """Set by SIGINT/SIGTERM (or programmatically) to request a graceful stop."""

    def __init__(self) -> None:
        self.event = threading.Event()
        self.reason = ""

    def request(self, reason: str) -> None:
        self.reason = reason
        self.event.set()

    def __bool__(self) -> bool:
        return self.event.is_set()


class Trainer:
    def __init__(
        self,
        config: RunConfig,
        run_dir: Path,
        *,
        resume: bool = False,
        log: Callable[[str], None] = print,
    ):
        self.config = config
        self.run_dir = run_dir
        self.log = log
        self.stop = _StopFlag()
        tc = config.train

        if tc.num_threads:
            torch.set_num_threads(tc.num_threads)
        self.device_info = select_device(tc.device)
        self.device = self.device_info.device
        self.amp_dtype = autocast_dtype(self.device, tc.precision)

        self.task = make_task(config, log)
        self.tokenizer = self.task.tokenizer
        mc = config.model

        self._seed_everything(tc.seed)
        self.model = TNSRModel(mc).to(self.device)
        fused = self.device.type == "cuda"
        self.optimizer = torch.optim.AdamW(
            self.model.optimizer_groups(tc.weight_decay),
            lr=tc.learning_rate,
            betas=(tc.beta1, tc.beta2),
            **({"fused": True} if fused else {}),
        )
        self.train_model: Any = self.model
        if tc.compile:
            self.train_model = torch.compile(self.model)
        self.data_rng = np.random.default_rng(tc.seed)
        self.step = 0
        self.elapsed_before = 0.0
        self.tokens_seen = 0
        self.best_val_loss: float | None = None
        self.last_val: dict[str, float] = {}
        self.last_extra: dict[str, float] = {}
        self.consecutive_nonfinite = 0
        self.nonfinite_total = 0
        self.peak_rss_mb = 0.0
        self.last_checkpoint: Path | None = None
        self.val_batches = self.task.eval_batches("val", tc.eval_iters, tc.batch_size)
        self.train_eval_batches = self.task.eval_batches("train", tc.eval_iters, tc.batch_size)
        if resume:
            self._restore_latest()

    # -- setup ------------------------------------------------------------------------

    @staticmethod
    def _seed_everything(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    @classmethod
    def create(
        cls, config: RunConfig, log: Callable[[str], None] = print, root: Path | None = None
    ) -> Trainer:
        config.validate()
        run_dir = new_run_dir(config.name, root)
        try:
            trainer = cls(config, run_dir, log=log)
        except Exception:
            shutil.rmtree(run_dir, ignore_errors=True)
            raise
        write_json_atomic(
            run_dir / "run.json",
            {
                "id": run_dir.name,
                "name": config.name,
                "task": config.task,
                "description": config.description,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "config": config.to_dict(),
                "num_parameters": trainer.model.num_parameters(),
                "num_parameters_non_embedding": trainer.model.num_parameters(non_embedding=True),
                "dataset": trainer._dataset_ref(),
                "tokenizer": trainer.tokenizer.describe(),
                "environment": environment_info(str(trainer.device), trainer.device_info.reason),
            },
        )
        return trainer

    @classmethod
    def resume_run(
        cls,
        run_dir: Path,
        log: Callable[[str], None] = print,
        max_steps: int | None = None,
        device: str | None = None,
    ) -> Trainer:
        run = read_run(run_dir)
        config = RunConfig.from_dict(run["config"])
        if max_steps is not None:
            config.train.max_steps = max_steps
        if device is not None:
            config.train.device = device
        config.validate()
        trainer = cls(config, run_dir, resume=True, log=log)
        if trainer.task.identity != run["dataset"]["identity"]:
            raise TrainingError(
                f"Dataset changed since this run started ({run['dataset']['identity']} -> "
                f"{trainer.task.identity}); resuming would mix data. Start a new run instead."
            )
        if max_steps is not None and max_steps != run["config"]["train"]["max_steps"]:
            run_json = {k: v for k, v in run.items() if k != "status"}
            run_json["config"]["train"]["max_steps"] = max_steps
            write_json_atomic(run_dir / "run.json", run_json)
        return trainer

    def _dataset_ref(self) -> dict[str, Any]:
        return self.task.dataset_ref()

    def _restore_latest(self) -> None:
        index = run_checkpoint_index(self.run_dir)
        latest = index.get("latest")
        if not latest:
            raise TrainingError(f"Run {self.run_dir.name} has no checkpoint to resume from")
        ckpt = self.run_dir / "checkpoints" / latest
        state_path = ckpt / "train_state.pt"
        if not state_path.is_file():
            raise TrainingError(f"Checkpoint {ckpt} has no train_state.pt; cannot resume")
        weights = torch.load(ckpt / "model.pt", map_location="cpu", weights_only=True)
        self.model.load_state_dict(weights)
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        if state["tokenizer_identity"] != self.tokenizer.identity:
            raise TrainingError("Tokenizer of the dataset differs from the checkpoint's tokenizer")
        self.optimizer.load_state_dict(state["optimizer"])
        self.step = state["step"]
        self.elapsed_before = state.get("elapsed", 0.0)
        self.tokens_seen = state.get("tokens_seen", 0)
        self.best_val_loss = state.get("best_val_loss")
        self.data_rng.bit_generator.state = state["data_rng"]
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"])
        if self.device.type == "cuda" and state.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        truncate_metrics_after(self.run_dir, self.step)
        self.log(f"Resumed {self.run_dir.name} from {latest} (step {self.step})")

    # -- evaluation -----------------------------------------------------------------------

    def _autocast(self):
        if self.amp_dtype is None:
            return contextlib.nullcontext()
        return torch.autocast(device_type=self.device.type, dtype=self.amp_dtype)

    @torch.no_grad()
    def estimate_loss(self) -> dict[str, float]:
        self.model.eval()
        out = {}
        for split, batches in (("train", self.train_eval_batches), ("val", self.val_batches)):
            losses = []
            for x, y in batches:
                x, y = x.to(self.device), y.to(self.device)
                with self._autocast():
                    _, loss = self.model(x, y)
                losses.append(loss.item())
            out[split] = float(np.mean(losses))
        self.model.train()
        return out

    # -- checkpointing ------------------------------------------------------------------

    def _rng_state(self) -> dict[str, Any]:
        return {
            "data_rng": self.data_rng.bit_generator.state,
            "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if self.device.type == "cuda" else None,
        }

    def save(self, elapsed: float) -> Path:
        name = f"step_{self.step:07d}"
        path = self.run_dir / "checkpoints" / name
        if path.exists():
            return path  # already saved at this step (e.g. final save right after an eval)
        tc = self.config.train
        meta = {
            "run_id": self.run_dir.name,
            "run_name": self.config.name,
            "step": self.step,
            "tokens_seen": self.tokens_seen,
            "task": self.config.task,
            "dataset": self._dataset_ref(),
            "training": {
                k: getattr(tc, k)
                for k in (
                    "seed",
                    "batch_size",
                    "grad_accum_steps",
                    "max_steps",
                    "learning_rate",
                    "min_lr",
                    "warmup_steps",
                    "lr_schedule",
                    "weight_decay",
                    "grad_clip",
                    "precision",
                )
            },
            "device": str(self.device),
            "metrics": {
                "train_loss": self.last_val.get("train"),
                "val_loss": self.last_val.get("val"),
                "val_perplexity": math.exp(self.last_val["val"])
                if "val" in self.last_val
                else None,
                "elapsed_seconds": round(elapsed, 2),
                **self.last_extra,
            },
        }
        train_state = {
            "step": self.step,
            "optimizer": self.optimizer.state_dict(),
            "elapsed": elapsed,
            "tokens_seen": self.tokens_seen,
            "best_val_loss": self.best_val_loss,
            "tokenizer_identity": self.tokenizer.identity,
            **self._rng_state(),
        }
        save_checkpoint(path, self.model, self.tokenizer, meta, train_state)
        self._update_index(name)
        self.last_checkpoint = path
        return path

    def _update_index(self, name: str) -> None:
        index = run_checkpoint_index(self.run_dir)
        entries = [e for e in index.get("checkpoints", []) if e["name"] != name]
        val = self.last_val.get("val")
        entries.append({"name": name, "step": self.step, "val_loss": val})
        entries.sort(key=lambda e: e["step"])
        best = index.get("best")
        best_entry = next((e for e in entries if e["name"] == best), None)
        if val is not None and (
            best_entry is None or best_entry["val_loss"] is None or val <= best_entry["val_loss"]
        ):
            best = name
        keep = {e["name"] for e in entries[-self.config.train.keep_checkpoints :]}
        if best:
            keep.add(best)
        for e in entries:
            if e["name"] not in keep:
                shutil.rmtree(self.run_dir / "checkpoints" / e["name"], ignore_errors=True)
        entries = [e for e in entries if e["name"] in keep]
        write_json_atomic(
            self.run_dir / "checkpoints.json",
            {"checkpoints": entries, "best": best, "latest": name},
        )

    # -- status -----------------------------------------------------------------------

    def _write_status(
        self, state: str, elapsed: float, extra: dict[str, Any] | None = None
    ) -> None:
        tc = self.config.train
        steps_done_this_session = max(1, self.step - self._session_start_step)
        session_elapsed = max(1e-9, elapsed - self.elapsed_before)
        rate = steps_done_this_session / session_elapsed
        remaining = max(0, tc.max_steps - self.step)
        write_json_atomic(
            self.run_dir / "status.json",
            {
                "state": state,
                "step": self.step,
                "max_steps": tc.max_steps,
                "progress": round(self.step / tc.max_steps, 4),
                "elapsed_seconds": round(elapsed, 1),
                "eta_seconds": round(remaining / rate, 1)
                if state == "running" and rate > 0
                else None,
                "tokens_seen": self.tokens_seen,
                "best_val_loss": self.best_val_loss,
                "last_val_loss": self.last_val.get("val"),
                "last_train_loss": self._last_loss,
                "peak_rss_mb": round(self.peak_rss_mb, 1),
                "device": str(self.device),
                "pid": psutil.Process().pid,
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "nonfinite_steps": self.nonfinite_total,
                **(extra or {}),
            },
        )

    # -- main loop ----------------------------------------------------------------------

    def install_signal_handlers(self) -> None:
        def handler(signum, _frame):
            self.stop.request(signal.Signals(signum).name)
            self.log(
                f"Received {signal.Signals(signum).name}; finishing step and saving checkpoint..."
            )

        signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGTERM, handler)

    def train(self) -> TrainResult:
        tc = self.config.train
        mc = self.config.model
        proc = psutil.Process()
        self._session_start_step = self.step
        self._last_loss: float | None = None
        t_start = time.time()
        self.model.train()
        tokens_per_step = tc.batch_size * tc.grad_accum_steps * mc.block_size
        self.log(
            f"Training {self.model.num_parameters():,} parameters on {self.device} "
            f"({self.device_info.reason}); {tokens_per_step:,} tokens/step, "
            f"steps {self.step}->{tc.max_steps}"
        )

        def elapsed() -> float:
            return self.elapsed_before + (time.time() - t_start)

        self._write_status("running", elapsed())
        status = "completed"
        window_t = time.time()
        window_tokens = 0
        try:
            while self.step < tc.max_steps:
                if self.stop:
                    status = "interrupted"
                    break
                lr = lr_at(self.step, tc)
                for group in self.optimizer.param_groups:
                    group["lr"] = lr
                self.optimizer.zero_grad(set_to_none=True)
                loss_sum = 0.0
                for _ in range(tc.grad_accum_steps):
                    x, y = self.task.train_batch(tc.batch_size, self.data_rng)
                    x, y = (
                        x.to(self.device, non_blocking=True),
                        y.to(self.device, non_blocking=True),
                    )
                    with self._autocast():
                        _, loss = self.train_model(x, y)
                    loss = loss / tc.grad_accum_steps
                    loss.backward()
                    loss_sum += loss.item()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), tc.grad_clip if tc.grad_clip > 0 else float("inf")
                ).item()
                if not (math.isfinite(loss_sum) and math.isfinite(grad_norm)):
                    self.consecutive_nonfinite += 1
                    self.nonfinite_total += 1
                    self.optimizer.zero_grad(set_to_none=True)
                    self.log(
                        f"step {self.step}: non-finite loss/gradient (loss={loss_sum}, grad_norm={grad_norm}); "
                        f"skipping update ({self.consecutive_nonfinite}/{tc.max_nonfinite})"
                    )
                    append_metrics(
                        self.run_dir,
                        {"type": "nonfinite", "step": self.step, "loss": str(loss_sum)},
                    )
                    if self.consecutive_nonfinite >= tc.max_nonfinite:
                        raise TrainingError(
                            f"Aborting: {self.consecutive_nonfinite} consecutive non-finite steps. "
                            "Try a lower learning_rate, enable grad_clip, or use fp32 precision."
                        )
                    continue
                self.consecutive_nonfinite = 0
                self.optimizer.step()
                self.step += 1
                self.tokens_seen += tokens_per_step
                window_tokens += tokens_per_step
                self._last_loss = loss_sum

                if self.step % tc.log_interval == 0 or self.step == tc.max_steps:
                    now = time.time()
                    tps = window_tokens / max(1e-9, now - window_t)
                    window_t, window_tokens = now, 0
                    self.peak_rss_mb = max(self.peak_rss_mb, proc.memory_info().rss / 1e6)
                    record = {
                        "type": "train",
                        "step": self.step,
                        "loss": round(loss_sum, 5),
                        "lr": lr,
                        "grad_norm": round(grad_norm, 4),
                        "tokens_per_sec": round(tps, 1),
                        "elapsed": round(elapsed(), 2),
                        "rss_mb": round(self.peak_rss_mb, 1),
                    }
                    if self.device.type == "cuda":
                        record["cuda_peak_mb"] = round(torch.cuda.max_memory_allocated() / 1e6, 1)
                    append_metrics(self.run_dir, record)
                    self.log(
                        f"step {self.step:>6}/{tc.max_steps} | loss {loss_sum:.4f} | lr {lr:.2e} | "
                        f"grad {grad_norm:.3f} | {tps:,.0f} tok/s"
                    )
                    self._write_status("running", elapsed())

                is_eval = self.step % tc.eval_interval == 0 or self.step == tc.max_steps
                if is_eval:
                    losses = self.estimate_loss()
                    extra = self.task.extra_eval(self.model, self.device)
                    self.last_extra = extra
                    self.last_val = losses
                    if self.best_val_loss is None or losses["val"] < self.best_val_loss:
                        self.best_val_loss = losses["val"]
                    append_metrics(
                        self.run_dir,
                        {
                            "type": "eval",
                            "step": self.step,
                            "train_loss": round(losses["train"], 5),
                            "val_loss": round(losses["val"], 5),
                            "val_perplexity": round(math.exp(min(losses["val"], 50)), 4),
                            "elapsed": round(elapsed(), 2),
                            **{k: round(v, 5) for k, v in extra.items()},
                        },
                    )
                    self.log(
                        f"eval @ {self.step}: train {losses['train']:.4f} | val {losses['val']:.4f} "
                        f"(ppl {math.exp(min(losses['val'], 50)):.2f})"
                        + "".join(f" | {k} {v:.4f}" for k, v in extra.items())
                    )
                ckpt_due = (
                    (tc.checkpoint_interval and self.step % tc.checkpoint_interval == 0)
                    or (not tc.checkpoint_interval and is_eval)
                    or self.step == tc.max_steps
                )
                if ckpt_due:
                    path = self.save(elapsed())
                    self.log(f"saved checkpoint {path.name}")
        except TrainingError as exc:
            self._write_status("failed", elapsed(), {"error": str(exc)})
            raise
        except Exception as exc:
            self._write_status("failed", elapsed(), {"error": f"{type(exc).__name__}: {exc}"})
            raise

        if status == "interrupted" and self.step > 0:
            if not self.last_val:
                self.last_val = self.estimate_loss()
            try:
                self.save(elapsed())
                self.log(
                    f"Interrupted at step {self.step}; checkpoint saved. Resume with: tnsr train --resume {self.run_dir.name}"
                )
            except CheckpointError as exc:
                self.log(f"Could not save interruption checkpoint: {exc}")
        self._write_status(status, elapsed(), {"stop_reason": self.stop.reason or None})
        self.log(f"Run {self.run_dir.name} {status} at step {self.step} in {elapsed():.1f}s")
        return TrainResult(
            self.run_dir, status, self.step, self.best_val_loss, self.last_checkpoint
        )

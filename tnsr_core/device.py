"""Compute device selection with graceful fallback."""

from __future__ import annotations

import platform
from dataclasses import dataclass

import torch


@dataclass
class DeviceInfo:
    device: torch.device
    requested: str
    reason: str

    @property
    def type(self) -> str:
        return self.device.type


def cuda_available() -> bool:
    return torch.cuda.is_available()


def mps_available() -> bool:
    backend = getattr(torch.backends, "mps", None)
    return bool(backend is not None and backend.is_available())


def select_device(requested: str = "auto") -> DeviceInfo:
    """Pick a device. ``auto`` prefers CUDA, then Apple MPS, then CPU.

    Explicit requests for an unavailable accelerator fall back to CPU with a
    reason instead of crashing, so configs stay portable across machines.
    """
    requested = (requested or "auto").lower()
    if requested not in {"auto", "cpu", "cuda", "mps"}:
        raise ValueError(f"Unknown device {requested!r}; expected auto, cpu, cuda or mps")
    if requested == "cpu":
        return DeviceInfo(torch.device("cpu"), requested, "CPU requested")
    if requested in {"auto", "cuda"} and cuda_available():
        name = torch.cuda.get_device_name(0)
        return DeviceInfo(torch.device("cuda"), requested, f"CUDA available ({name})")
    if requested in {"auto", "mps"} and mps_available():
        return DeviceInfo(torch.device("mps"), requested, "Apple Metal (MPS) available")
    if requested == "auto":
        return DeviceInfo(torch.device("cpu"), requested, "No accelerator detected; using CPU")
    return DeviceInfo(
        torch.device("cpu"),
        requested,
        f"{requested.upper()} requested but unavailable; falling back to CPU",
    )


# Below this size, single-token decoding on Apple MPS is dominated by kernel-launch overhead:
# measured on an Apple Silicon MacBook Air, a 0.8M model decoded at ~3,000 tok/s on CPU vs
# ~850 tok/s on MPS. Training (4K-token batches) is still faster on MPS, so this applies to
# inference only.
MPS_INFERENCE_MIN_PARAMS = 50_000_000


def select_inference_device(
    requested: str = "auto", num_parameters: int | None = None
) -> DeviceInfo:
    """Like :func:`select_device`, but ``auto`` keeps small models off MPS for generation."""
    info = select_device(requested)
    if (
        info.requested == "auto"
        and info.device.type == "mps"
        and num_parameters is not None
        and num_parameters < MPS_INFERENCE_MIN_PARAMS
    ):
        return DeviceInfo(
            torch.device("cpu"),
            requested,
            f"CPU chosen over MPS for a small model ({num_parameters / 1e6:.1f}M parameters): "
            "per-token GPU overhead makes MPS slower for generation at this size",
        )
    return info


def autocast_dtype(device: torch.device, precision: str) -> torch.dtype | None:
    """Return the autocast dtype for ``precision`` on ``device`` (None = full fp32)."""
    precision = precision.lower()
    if precision == "fp32":
        return None
    if precision not in {"auto", "bf16", "fp16"}:
        raise ValueError(f"Unknown precision {precision!r}")
    if device.type == "cuda":
        if precision == "fp16":
            return torch.float16
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16 if precision == "auto" else None
    if precision == "auto":
        # CPU bf16 autocast is slower than fp32 on most consumer CPUs and MPS
        # autocast coverage varies by macOS version: keep fp32 by default.
        return None
    if device.type == "cpu" and precision == "bf16":
        return torch.bfloat16
    return None


def system_summary() -> dict:
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": cuda_available(),
        "mps_available": mps_available(),
        "cpu_threads": torch.get_num_threads(),
    }

"""Environment diagnostics (``tnsr doctor``)."""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time
from typing import Any

import psutil
import torch

from tnsr_core import __version__
from tnsr_core.device import cuda_available, mps_available, select_device
from tnsr_core.paths import datasets_dir, experiments_dir, tnsr_home, ui_dist_dir


def _check(name: str, ok: bool, detail: str, level: str = "error") -> dict[str, Any]:
    return {"name": name, "status": "ok" if ok else level, "detail": detail}


def run_diagnostics(quick: bool = False) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    py_ok = sys.version_info >= (3, 10)
    checks.append(_check("python", py_ok, f"Python {sys.version.split()[0]} (requires >= 3.10)"))
    checks.append(_check("torch", True, f"PyTorch {torch.__version__}"))

    info = select_device("auto")
    checks.append(_check("device", True, f"{info.device} - {info.reason}"))
    if cuda_available():
        checks.append(_check("cuda", True, torch.cuda.get_device_name(0)))
    if sys.platform == "darwin":
        checks.append(
            _check(
                "mps",
                mps_available(),
                "Apple Metal available"
                if mps_available()
                else "MPS unavailable (Intel Mac or old macOS); CPU will be used",
                level="warning",
            )
        )

    mem = psutil.virtual_memory()
    checks.append(
        _check(
            "memory",
            mem.available > 2e9,
            f"{mem.available / 1e9:.1f} GB available of {mem.total / 1e9:.1f} GB",
            level="warning",
        )
    )
    home = tnsr_home()
    try:
        disk = shutil.disk_usage(home)
        checks.append(
            _check(
                "disk", disk.free > 2e9, f"{disk.free / 1e9:.1f} GB free at {home}", level="warning"
            )
        )
    except OSError as exc:
        checks.append(_check("disk", False, str(exc), level="warning"))
    try:
        with tempfile.NamedTemporaryFile(dir=home, prefix=".tnsr-write-test"):
            pass
        checks.append(_check("home_writable", True, f"TNSR_HOME={home}"))
    except OSError as exc:
        checks.append(_check("home_writable", False, f"Cannot write to {home}: {exc}"))

    node = shutil.which("node")
    checks.append(
        _check(
            "node",
            node is not None,
            f"Node.js at {node}" if node else "Node.js not found (needed only to build the UI)",
            level="warning",
        )
    )
    ui_built = (ui_dist_dir() / "index.html").is_file()
    checks.append(
        _check(
            "ui_build",
            ui_built,
            "UI build present" if ui_built else "UI not built (run: npm run build)",
            level="warning",
        )
    )

    n_datasets = (
        len([p for p in datasets_dir().glob("*/meta.json")]) if datasets_dir().is_dir() else 0
    )
    n_runs = (
        len([p for p in experiments_dir().glob("*/run.json")]) if experiments_dir().is_dir() else 0
    )
    checks.append(
        _check(
            "datasets",
            n_datasets > 0,
            f"{n_datasets} prepared dataset(s)"
            + ("" if n_datasets else " - run: tnsr data prepare"),
            level="warning",
        )
    )
    checks.append(_check("experiments", True, f"{n_runs} training run(s)"))

    if not quick:
        from tnsr_core.config import ModelConfig
        from tnsr_core.model import TNSRModel

        try:
            t0 = time.perf_counter()
            model = TNSRModel(
                ModelConfig(vocab_size=64, block_size=16, n_layer=1, n_head=2, n_embd=16)
            ).to(info.device)
            x = torch.randint(0, 64, (2, 16), device=info.device)
            _, loss = model(x, x)
            loss.backward()
            checks.append(
                _check(
                    "model_smoke",
                    True,
                    f"forward+backward on {info.device} in {(time.perf_counter() - t0) * 1000:.0f} ms",
                )
            )
        except Exception as exc:  # report, don't crash
            checks.append(
                _check(
                    "model_smoke",
                    False,
                    f"{type(exc).__name__}: {exc}. Try --device cpu, or set PYTORCH_ENABLE_MPS_FALLBACK=1 on macOS.",
                )
            )

    return {
        "tnsr_version": __version__,
        "tnsr_home": str(home),
        "selected_device": str(info.device),
        "cpu_count": os.cpu_count(),
        "torch_threads": torch.get_num_threads(),
        "checks": checks,
        "ok": all(c["status"] != "error" for c in checks),
    }

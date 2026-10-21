"""Filesystem layout for TNSR artifacts.

All generated artifacts live under a single home directory (``TNSR_HOME``,
defaulting to the current working directory):

    experiments/<run_id>/       training runs (config, metrics, checkpoints, evals)
    datasets/<name>/            prepared token datasets and their tokenizers
    checkpoints/<name>/         imported / exported standalone checkpoints
    var/                        local database, job logs (never committed)
"""

from __future__ import annotations

import os
import re
from pathlib import Path

_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def tnsr_home() -> Path:
    return Path(os.environ.get("TNSR_HOME", os.getcwd())).resolve()


def experiments_dir() -> Path:
    return tnsr_home() / "experiments"


def datasets_dir() -> Path:
    return tnsr_home() / "datasets"


def checkpoints_dir() -> Path:
    return tnsr_home() / "checkpoints"


def var_dir() -> Path:
    return tnsr_home() / "var"


def validate_name(name: str, what: str = "name") -> str:
    """Validate a user-supplied artifact name (no path separators, no traversal)."""
    if not _SAFE_NAME.match(name) or ".." in name:
        raise ValueError(
            f"Invalid {what} {name!r}: use letters, digits, '.', '_' or '-' (max 128 chars)."
        )
    return name


def safe_join(root: Path, *parts: str) -> Path:
    """Join ``parts`` under ``root`` and refuse any result that escapes ``root``."""
    root = root.resolve()
    candidate = root.joinpath(*parts).resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError(f"Path {candidate} escapes {root}")
    return candidate


def repo_root() -> Path:
    """Source checkout root (parent of the tnsr_core package)."""
    return Path(__file__).resolve().parent.parent


def ui_dist_dir() -> Path:
    return Path(os.environ.get("TNSR_UI_DIST", repo_root() / "tnsr_ui" / "dist")).resolve()

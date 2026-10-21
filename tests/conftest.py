from __future__ import annotations

from pathlib import Path

import pytest
import torch

from tnsr_core.config import load_config
from tnsr_core.data import prepare_dataset
from tnsr_core.data.synthetic import generate_documents

torch.set_num_threads(2)


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolated TNSR_HOME so tests never touch real artifacts."""
    monkeypatch.setenv("TNSR_HOME", str(tmp_path))
    return tmp_path


def make_dataset(name: str = "demo", kind: str = "bpe", n_docs: int = 400, vocab_size: int = 400):
    return prepare_dataset(
        name,
        list(generate_documents(n_docs, seed=0)),
        tokenizer_kind=kind,
        vocab_size=vocab_size,
        val_fraction=0.1,
    )


@pytest.fixture
def dataset(home: Path):
    return make_dataset()


def tiny_config(**train_overrides):
    overrides = {f"train.{k}": v for k, v in train_overrides.items()}
    return load_config("tiny", overrides)


@pytest.fixture(scope="session")
def trained_home(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A home with a dataset and a short completed tiny run (shared, read-only use)."""
    import os

    from tnsr_core.training import Trainer

    root = tmp_path_factory.mktemp("trained")
    old = os.environ.get("TNSR_HOME")
    os.environ["TNSR_HOME"] = str(root)
    try:
        make_dataset()
        cfg = tiny_config(max_steps=60, eval_interval=30, learning_rate=1e-2)
        Trainer.create(cfg, log=lambda _m: None).train()
    finally:
        if old is None:
            os.environ.pop("TNSR_HOME", None)
        else:
            os.environ["TNSR_HOME"] = old
    return root


@pytest.fixture
def trained(trained_home: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """TNSR_HOME pointing at the shared trained home; returns the best checkpoint path."""
    from tnsr_core.checkpoints import best_checkpoint_of_run

    monkeypatch.setenv("TNSR_HOME", str(trained_home))
    run = next((trained_home / "experiments").iterdir())
    return best_checkpoint_of_run(run)

import json
import math

import pytest
import torch

from tnsr_core.checkpoints import load_checkpoint, run_checkpoint_index
from tnsr_core.config import ConfigError, RunConfig, TrainConfig, load_config, parse_override
from tnsr_core.device import select_device
from tnsr_core.training import Trainer, TrainingError, lr_at, read_metrics, read_run

from .conftest import tiny_config


def quiet(_msg):
    pass


def test_smoke_training_reduces_loss_and_writes_artifacts(dataset):
    cfg = tiny_config(max_steps=40, eval_interval=20, learning_rate=1e-2)
    result = Trainer.create(cfg, log=quiet).train()
    assert result.status == "completed" and result.step == 40
    run = result.run_dir
    metrics = read_metrics(run)
    evals = [m for m in metrics if m["type"] == "eval"]
    trains = [m for m in metrics if m["type"] == "train"]
    assert [e["step"] for e in evals] == [20, 40]
    assert trains[-1]["loss"] < trains[0]["loss"]
    assert all(math.isfinite(e["val_loss"]) for e in evals)
    status = json.loads((run / "status.json").read_text())
    assert status["state"] == "completed" and status["step"] == 40
    info = read_run(run)
    assert info["dataset"]["identity"] == dataset.meta["identity"]
    assert info["environment"]["torch"] == torch.__version__
    index = run_checkpoint_index(run)
    assert index["latest"] == "step_0000040"
    ckpt = run / "checkpoints" / "step_0000040"
    meta = json.loads((ckpt / "meta.json").read_text())
    for key in ("model_config", "tokenizer", "step", "dataset", "training", "metrics"):
        assert key in meta
    assert meta["tokenizer"]["identity"] == dataset.meta["tokenizer"]["identity"]
    loaded = load_checkpoint(ckpt)
    assert loaded.model.num_parameters() == meta["num_parameters"]


def test_resume_is_exactly_equivalent_to_uninterrupted_training(dataset):
    straight = Trainer.create(tiny_config(max_steps=20, eval_interval=10), log=quiet)
    straight.train()

    first = Trainer.create(tiny_config(max_steps=20, eval_interval=10), log=quiet)
    original = first.optimizer.step

    def step_and_stop(*a, **kw):
        out = original(*a, **kw)
        if first.step + 1 >= 10:  # step counter increments right after optimizer.step()
            first.stop.request("test")
        return out

    first.optimizer.step = step_and_stop
    assert first.train().status == "interrupted"
    resumed = Trainer.resume_run(first.run_dir, log=quiet)
    assert resumed.step == 10
    resumed.train()

    a = straight.model.state_dict()
    b = resumed.model.state_dict()
    for k in a:
        assert torch.equal(a[k], b[k]), k
    # metrics after resume continue from the checkpoint step without duplicates
    steps = [m["step"] for m in read_metrics(first.run_dir) if m["type"] == "train"]
    assert steps == sorted(set(steps))
    extended = Trainer.resume_run(first.run_dir, log=quiet, max_steps=25)
    extended.train()
    assert read_run(first.run_dir)["config"]["train"]["max_steps"] == 25


def test_keeps_limited_checkpoints_plus_best(dataset):
    cfg = tiny_config(max_steps=40, eval_interval=10, keep_checkpoints=2)
    result = Trainer.create(cfg, log=quiet).train()
    names = sorted(p.name for p in (result.run_dir / "checkpoints").iterdir())
    index = run_checkpoint_index(result.run_dir)
    assert set(names) == {e["name"] for e in index["checkpoints"]}
    assert len(names) <= 3 and index["best"] in names and "step_0000040" in names


def test_graceful_interrupt_saves_checkpoint(dataset):
    trainer = Trainer.create(tiny_config(max_steps=1000, eval_interval=500), log=quiet)
    original = trainer.optimizer.step

    def step_and_stop(*a, **kw):
        out = original(*a, **kw)
        if trainer.step >= 6:
            trainer.stop.request("test")
        return out

    trainer.optimizer.step = step_and_stop
    result = trainer.train()
    assert result.status == "interrupted"
    assert result.last_checkpoint is not None and result.last_checkpoint.name == "step_0000007"
    status = json.loads((trainer.run_dir / "status.json").read_text())
    assert status["state"] == "interrupted"
    resumed = Trainer.resume_run(trainer.run_dir, log=quiet)
    assert resumed.step == 7


def test_nonfinite_loss_aborts_with_clear_error(dataset, monkeypatch):
    trainer = Trainer.create(tiny_config(max_steps=10, max_nonfinite=2), log=quiet)
    real_forward = trainer.model.forward

    def nan_forward(idx, targets=None, **kw):
        logits, loss = real_forward(idx, targets, **kw)
        return logits, loss * float("nan") if loss is not None else None

    monkeypatch.setattr(trainer, "train_model", nan_forward)
    with pytest.raises(TrainingError, match="non-finite"):
        trainer.train()
    status = json.loads((trainer.run_dir / "status.json").read_text())
    assert status["state"] == "failed"
    assert trainer.step == 0


def test_invalid_configs():
    with pytest.raises(ConfigError):
        TrainConfig(batch_size=0).validate()
    with pytest.raises(ConfigError):
        TrainConfig(min_lr=1.0, learning_rate=0.1).validate()
    with pytest.raises(ConfigError, match="Unknown keys"):
        RunConfig.from_dict({"train": {"batchsize": 3}})
    with pytest.raises(ConfigError, match="must be int"):
        RunConfig.from_dict({"train": {"batch_size": "3"}})
    with pytest.raises(ConfigError, match="not found"):
        load_config("does-not-exist")


def test_overrides_parse_toml_values():
    assert parse_override("train.max_steps=5") == ("train.max_steps", 5)
    assert parse_override("model.norm=layernorm") == ("model.norm", "layernorm")
    assert parse_override("train.learning_rate=1e-3") == ("train.learning_rate", 1e-3)
    cfg = load_config("tiny", dict([parse_override("train.max_steps=7")]))
    assert cfg.train.max_steps == 7


def test_missing_dataset_fails_fast_without_leaving_a_run(home):
    from tnsr_core.data import DatasetError

    with pytest.raises(DatasetError):
        Trainer.create(tiny_config(), log=quiet)
    assert not list((home / "experiments").glob("*"))


def test_device_fallback():
    assert select_device("cpu").device.type == "cpu"
    if not torch.backends.mps.is_available():
        info = select_device("mps")
        assert info.device.type == "cpu" and "falling back" in info.reason
    if not torch.cuda.is_available():
        assert select_device("cuda").device.type == "cpu"
    with pytest.raises(ValueError):
        select_device("tpu")


def test_lr_schedule():
    cfg = TrainConfig(max_steps=100, warmup_steps=10, learning_rate=1.0, min_lr=0.1)
    assert lr_at(0, cfg) == pytest.approx(0.1)
    assert lr_at(9, cfg) == pytest.approx(1.0)
    assert lr_at(100, cfg) == pytest.approx(0.1)
    values = [lr_at(s, cfg) for s in range(10, 100)]
    assert values == sorted(values, reverse=True)
    cfg.lr_schedule = "constant"
    assert lr_at(50, cfg) == 1.0


def test_small_models_generate_on_cpu_instead_of_mps(monkeypatch):
    from tnsr_core import device as dev

    monkeypatch.setattr(dev, "cuda_available", lambda: False)
    monkeypatch.setattr(dev, "mps_available", lambda: True)
    small = dev.select_inference_device("auto", 3_000_000)
    assert small.device.type == "cpu" and "small model" in small.reason
    assert dev.select_inference_device("auto", 200_000_000).device.type == "mps"
    # An explicit request is always honoured; training keeps using MPS under auto.
    assert dev.select_inference_device("mps", 3_000_000).device.type == "mps"
    assert dev.select_device("auto").device.type == "mps"

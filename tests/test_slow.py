"""Longer training/benchmark checks. Run with: uv run pytest -m slow"""

import math

import pytest

from tnsr_core.config import load_config
from tnsr_core.evaluation import benchmark_checkpoint, evaluate_checkpoint
from tnsr_core.inference import GenerationConfig, InferenceEngine
from tnsr_core.training import Trainer

from .conftest import make_dataset

pytestmark = pytest.mark.slow


def test_small_model_learns_the_synthetic_corpus(home):
    make_dataset(n_docs=3000, vocab_size=800)
    cfg = load_config(
        "tiny",
        {
            "model.block_size": 128,
            "model.n_layer": 3,
            "model.n_head": 4,
            "model.n_embd": 96,
            "train.max_steps": 600,
            "train.batch_size": 16,
            "train.eval_interval": 200,
            "train.learning_rate": 3e-3,
            "train.warmup_steps": 30,
        },
    )
    result = Trainer.create(cfg, log=lambda _m: None).train()
    assert result.best_val_loss is not None
    # Uniform guessing over ~800 tokens is ln(800) ≈ 6.7 nats; the templated corpus is far more predictable.
    assert result.best_val_loss < 2.0, result.best_val_loss
    ckpt = result.last_checkpoint
    assert ckpt is not None
    ev = evaluate_checkpoint(ckpt, device="cpu", sample_prompts=[], save=False)
    assert ev["metrics"]["bits_per_byte"] < 1.0
    assert math.isclose(
        ev["metrics"]["perplexity"], math.exp(ev["metrics"]["val_loss"]), rel_tol=1e-3
    )

    engine = InferenceEngine.from_checkpoint(ckpt, "cpu")
    text = engine.generate(
        "One day, a", GenerationConfig(max_new_tokens=40, deterministic=True)
    ).text
    assert len(text.split()) >= 5  # produces word-like continuations, not byte noise


def test_kv_cache_speeds_up_long_generation(home):
    make_dataset(n_docs=600, vocab_size=400)
    cfg = load_config(
        "tiny",
        {
            "model.block_size": 256,
            "model.n_layer": 4,
            "model.n_embd": 128,
            "model.n_head": 4,
            "train.max_steps": 10,
        },
    )
    result = Trainer.create(cfg, log=lambda _m: None).train()
    assert result.last_checkpoint is not None
    bench = benchmark_checkpoint(
        result.last_checkpoint,
        device="cpu",
        prompt_tokens=(64,),
        new_tokens=160,
        repeats=3,
        save=False,
    )
    rows = {r["kv_cache"]: r for r in bench["results"]}
    assert rows[True]["decode_tokens_per_sec_median"] > rows[False]["decode_tokens_per_sec_median"]

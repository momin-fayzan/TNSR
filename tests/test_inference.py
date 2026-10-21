import json
import shutil
import threading

import pytest
import torch

from tnsr_core.checkpoints import (
    CheckpointError,
    discover_checkpoints,
    export_checkpoint,
    import_checkpoint,
    load_checkpoint,
    resolve_checkpoint_id,
)
from tnsr_core.inference import GenerationConfig, GenerationConfigError, InferenceEngine
from tnsr_core.inference.engine import _StopMatcher
from tnsr_core.inference.sampling import filter_logits, select_token


@pytest.fixture
def engine(trained):
    return InferenceEngine.from_checkpoint(trained, "cpu")


def test_generates_requested_number_of_tokens(engine):
    res = engine.generate(
        "Once upon a time", GenerationConfig(max_new_tokens=12, stop_at_eos=False, seed=0)
    )
    assert res.stats.generated_tokens == 12
    assert res.stats.finish_reason == "length"
    assert res.text
    assert res.stats.time_to_first_token_ms is not None and res.stats.time_to_first_token_ms > 0


def test_greedy_generation_is_deterministic(engine):
    cfg = GenerationConfig(max_new_tokens=20, deterministic=True)
    a = engine.generate("One day", cfg).text
    b = engine.generate("One day", cfg).text
    assert a == b


def test_seeded_sampling_is_reproducible(engine):
    cfg = GenerationConfig(max_new_tokens=20, temperature=1.0, seed=123, stop_at_eos=False)
    a = engine.generate("One day", cfg).text
    b = engine.generate("One day", cfg).text
    c = engine.generate(
        "One day", GenerationConfig(max_new_tokens=20, temperature=1.0, seed=124, stop_at_eos=False)
    ).text
    assert a == b
    assert a != c  # overwhelmingly likely for an untrained-ish model at T=1


def test_kv_cache_matches_uncached_generation(engine):
    cfg = GenerationConfig(max_new_tokens=40, deterministic=True, stop_at_eos=False)
    cached = engine.generate("The cat", cfg, use_cache=True)
    uncached = engine.generate("The cat", cfg, use_cache=False)
    assert cached.token_ids == uncached.token_ids


def test_generation_beyond_context_window(engine):
    block = engine.block_size
    cfg = GenerationConfig(max_new_tokens=block * 2, deterministic=True, stop_at_eos=False)
    res = engine.generate("A", cfg)
    assert res.stats.generated_tokens == block * 2
    assert res.stats.cache_rebuilds >= 1


def test_long_prompt_is_truncated(engine):
    res = engine.generate("word " * 500, GenerationConfig(max_new_tokens=2, deterministic=True))
    assert res.stats.prompt_truncated
    assert res.stats.prompt_tokens == engine.block_size - 1


def test_stops_at_eos(engine):
    eos = engine.tokenizer.eos_id
    original = engine.model.forward

    def forced_eos(*a, **kw):
        logits, loss = original(*a, **kw)
        logits = torch.full_like(logits, -1e9)
        logits[..., eos] = 0
        return logits, loss

    engine.model.forward = forced_eos
    try:
        res = engine.generate("hi", GenerationConfig(max_new_tokens=10, deterministic=True))
    finally:
        engine.model.forward = original
    assert res.stats.finish_reason == "eos"
    assert res.text == "" and res.stats.generated_tokens == 0


def test_stop_strings_are_excluded_and_never_partially_streamed():
    m = _StopMatcher(["\nUser:"])
    out = []
    hit = False
    for piece in ["Hello", " there", "\n", "Us", "er:", " more"]:
        text, hit = m.push(piece)
        out.append(text)
        if hit:
            break
    assert hit and "".join(out) == "Hello there"
    assert "\n" not in "".join(out)
    m = _StopMatcher(["\nUser:"])
    assert m.push("a\nU")[0] == "a"
    assert m.push("nicorn")[0] == "\nUnicorn"


def test_streaming_concatenation_equals_generate(engine):
    cfg = GenerationConfig(max_new_tokens=25, deterministic=True)
    events = list(engine.stream("Once", cfg))
    assert events[-1].kind == "done"
    streamed = "".join(e.text for e in events if e.kind == "token")
    assert streamed == engine.generate("Once", cfg).text


def test_cancellation_stops_generation(engine):
    cancel = threading.Event()
    seen = 0
    final = None
    for event in engine.stream(
        "Once",
        GenerationConfig(max_new_tokens=500, deterministic=True, stop_at_eos=False),
        cancel=cancel,
    ):
        if event.kind == "token":
            seen += 1
            if seen == 3:
                cancel.set()
        else:
            final = event.stats
    assert final is not None and final.finish_reason == "cancelled"
    assert final.generated_tokens < 10


@pytest.mark.parametrize(
    "kw",
    [
        {"max_new_tokens": 0},
        {"max_new_tokens": 10**6},
        {"temperature": -1},
        {"top_k": -1},
        {"top_p": 0},
        {"top_p": 1.5},
        {"repetition_penalty": 0.5},
        {"stop": [""]},
    ],
)
def test_invalid_generation_parameters(kw):
    with pytest.raises(GenerationConfigError):
        GenerationConfig(**kw).validate(vocab_size=100)


def test_top_k_larger_than_vocab_rejected():
    with pytest.raises(GenerationConfigError):
        GenerationConfig(top_k=101).validate(vocab_size=100)


def test_top_k_and_top_p_filtering():
    logits = torch.tensor([1.0, 3.0, 2.0, 0.0])
    k = filter_logits(logits, top_k=2, top_p=1.0)
    assert torch.isinf(k[[0, 3]]).all() and torch.isfinite(k[[1, 2]]).all()
    # probs ~ [0.09, 0.67, 0.24, 0.03]; top_p=0.5 keeps only the argmax
    p = filter_logits(logits, top_k=0, top_p=0.5)
    assert torch.isfinite(p).sum() == 1 and torch.isfinite(p[1])
    p = filter_logits(logits, top_k=0, top_p=0.8)
    assert set(torch.nonzero(torch.isfinite(p)).flatten().tolist()) == {1, 2}


def test_top_k_one_equals_greedy():
    logits = torch.randn(50)
    g = torch.Generator().manual_seed(0)
    cfg = GenerationConfig(temperature=1.5, top_k=1)
    assert select_token(logits, cfg, torch.tensor([]), g) == int(torch.argmax(logits))


def test_checkpoint_listing_and_resolution(trained):
    found = discover_checkpoints()
    assert any(c["is_best"] for c in found)
    best = next(c for c in found if c["is_best"])
    assert resolve_checkpoint_id(best["id"]) == trained
    assert resolve_checkpoint_id(best["run_id"]) == trained
    for bad in ["../etc", "experiments/../../x", "/etc/passwd", ""]:
        with pytest.raises((CheckpointError, ValueError)):
            resolve_checkpoint_id(bad)


def _copy(trained, tmp_path):
    dst = tmp_path / "ckpt"
    shutil.copytree(trained, dst)
    return dst


def test_missing_checkpoint_files(trained, tmp_path):
    with pytest.raises(CheckpointError, match="not found"):
        load_checkpoint(tmp_path / "nope")
    ck = _copy(trained, tmp_path)
    (ck / "model.pt").unlink()
    with pytest.raises(CheckpointError, match="missing model.pt"):
        load_checkpoint(ck)


def test_incompatible_tokenizer_is_rejected(trained, tmp_path):
    from tnsr_core.tokenizer import CharTokenizer, save_tokenizer

    ck = _copy(trained, tmp_path)
    save_tokenizer(CharTokenizer.train(["abc"]), ck / "tokenizer.json")
    with pytest.raises(CheckpointError, match="Tokenizer mismatch"):
        load_checkpoint(ck)


def test_vocab_mismatch_is_rejected(trained, tmp_path):
    ck = _copy(trained, tmp_path)
    meta = json.loads((ck / "meta.json").read_text())
    meta["model_config"]["vocab_size"] += 1
    (ck / "meta.json").write_text(json.dumps(meta))
    with pytest.raises(CheckpointError, match="Vocabulary size mismatch"):
        load_checkpoint(ck)


def test_shape_mismatch_and_future_format_rejected(trained, tmp_path):
    ck = _copy(trained, tmp_path)
    meta = json.loads((ck / "meta.json").read_text())
    meta["model_config"]["n_layer"] += 1
    (ck / "meta.json").write_text(json.dumps(meta))
    with pytest.raises(CheckpointError, match="do not match"):
        load_checkpoint(ck)
    meta["format_version"] = 999
    (ck / "meta.json").write_text(json.dumps(meta))
    with pytest.raises(CheckpointError, match="format version"):
        load_checkpoint(ck)


def test_inference_does_not_need_dataset_or_optimizer_state(trained, tmp_path):
    ck = _copy(trained, tmp_path)
    (ck / "train_state.pt").unlink()
    engine = InferenceEngine.from_checkpoint(ck, "cpu")
    assert (
        engine.generate("Hi", GenerationConfig(max_new_tokens=3, deterministic=True)).stats.total_ms
        > 0
    )


def test_export_import_roundtrip(trained, tmp_path):
    archive = export_checkpoint(trained, tmp_path / "model.tar.gz")
    with pytest.raises(CheckpointError):
        export_checkpoint(trained, archive)  # never overwrite
    dest = import_checkpoint(archive, "imported")
    a = load_checkpoint(trained).model.state_dict()
    b = load_checkpoint(dest).model.state_dict()
    assert all(torch.equal(a[k], b[k]) for k in a)
    assert any(c["id"] == "checkpoints/imported" for c in discover_checkpoints())
    with pytest.raises(CheckpointError, match="already exists"):
        import_checkpoint(archive, "imported")
    shutil.rmtree(dest)

import numpy as np
import pytest

from tnsr_core.data import (
    DatasetError,
    TokenDataset,
    prepare_dataset,
    read_text_documents,
    split_documents,
)
from tnsr_core.data.synthetic import generate_documents


def test_synthetic_corpus_is_deterministic():
    assert list(generate_documents(50, seed=3)) == list(generate_documents(50, seed=3))
    assert list(generate_documents(50, seed=3)) != list(generate_documents(50, seed=4))


def test_synthetic_arithmetic_is_correct():
    import re

    for doc in generate_documents(2000, seed=1):
        for a, b, c in re.findall(r"(\d+) plus (\d+) is (\d+)\.", doc):
            assert int(a) + int(b) == int(c)
        for a, b, c in re.findall(r"(\d+) minus (\d+) is (\d+)\.", doc):
            assert int(a) - int(b) == int(c)


def test_split_is_reproducible_and_leak_free():
    docs = [f"document number {i}" for i in range(500)] + ["duplicate"] * 20
    train, val = split_documents(docs, 0.2, seed=7)
    train2, val2 = split_documents(list(reversed(docs)), 0.2, seed=7)
    assert set(train) == set(train2) and set(val) == set(val2)
    assert not set(train) & set(val)
    assert 50 < len(val) < 160
    # all copies of a duplicated document land in the same split
    assert ("duplicate" in train) != ("duplicate" in val)


def test_split_requires_both_sides():
    with pytest.raises(DatasetError):
        split_documents(["only one"], 0.5, seed=0)
    with pytest.raises(DatasetError):
        split_documents(["a", "b"], 1.5, seed=0)


def test_tokenizer_is_fit_on_train_split_only(home):
    docs = [f"common text {i}" for i in range(200)]
    train, val = split_documents(docs + ["ZZZ unique"], 0.1, seed=0)
    marker = "ZZZ unique"
    if marker in train:  # pick a seed that places the marker in validation
        for seed in range(1, 50):
            train, val = split_documents(docs + [marker], 0.1, seed=seed)
            if marker in val:
                break
        else:
            pytest.skip("could not place marker in val")
    else:
        seed = 0
    ds = prepare_dataset(
        "leak", docs + [marker], tokenizer_kind="char", val_fraction=0.1, seed=seed
    )
    loaded = TokenDataset("leak")
    assert "Z" not in loaded.tokenizer.chars
    assert ds.meta["stats"]["train_documents"] + ds.meta["stats"]["val_documents"] == 201


def test_prepare_writes_files_and_stats(dataset):
    path = dataset.path
    for name in ("train.bin", "val.bin", "tokenizer.json", "meta.json"):
        assert (path / name).is_file()
    ds = TokenDataset("demo")
    stats = dataset.meta["stats"]
    assert ds.num_tokens("train") == stats["train_tokens"]
    assert ds.num_tokens("val") == stats["val_tokens"]
    eos = ds.tokenizer.eos_id
    # one EOS per document
    assert int((np.asarray(ds.splits["train"]) == eos).sum()) == stats["train_documents"]


def test_prepare_refuses_to_overwrite(dataset):
    with pytest.raises(DatasetError, match="already exists"):
        prepare_dataset(
            "demo", ["a b c"] * 50 + ["d e f"] * 50, tokenizer_kind="char", val_fraction=0.3
        )


def test_invalid_dataset_name(home):
    with pytest.raises(ValueError):
        prepare_dataset("../escape", ["x"] * 10, tokenizer_kind="char")


def test_batches_are_shifted_windows(dataset):
    ds = TokenDataset("demo")
    rng = np.random.default_rng(0)
    x, y = ds.random_batch("train", 4, 16, rng)
    assert x.shape == (4, 16) and y.shape == (4, 16)
    assert torch_equal(x[:, 1:], y[:, :-1])
    fixed1 = ds.fixed_batches("val", 2, 3, 16)
    fixed2 = ds.fixed_batches("val", 2, 3, 16)
    assert all(torch_equal(a[0], b[0]) for a, b in zip(fixed1, fixed2, strict=True))


def test_sequential_batches_cover_split(dataset):
    ds = TokenDataset("demo")
    total = sum(y.numel() for _, y in ds.sequential_batches("val", 16, 4))
    assert ds.num_tokens("val") - 1 - 16 < total <= ds.num_tokens("val") - 1


def test_block_size_check(dataset):
    ds = TokenDataset("demo")
    with pytest.raises(DatasetError, match="block_size"):
        ds.check_block_size(10**7)


def test_read_text_documents(tmp_path):
    (tmp_path / "a.txt").write_text("para one\n\npara two\n\n\n")
    (tmp_path / "b.md").write_text("line a\nline b")
    (tmp_path / "skip.bin").write_bytes(b"\x00")
    docs, info = read_text_documents(tmp_path, "paragraph")
    assert docs == ["para one", "para two", "line a\nline b"]
    assert info["files"] == 2
    docs, _ = read_text_documents(tmp_path, "line")
    assert "line b" in docs
    with pytest.raises(DatasetError):
        read_text_documents(tmp_path / "missing")


def test_unknown_dataset_message(home):
    with pytest.raises(DatasetError, match="tnsr data prepare"):
        TokenDataset("nope")


def torch_equal(a, b):
    import torch

    return torch.equal(a, b)

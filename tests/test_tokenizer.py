import json

import pytest

from tnsr_core.tokenizer import (
    EOS,
    BPETokenizer,
    CharTokenizer,
    load_tokenizer,
    save_tokenizer,
    tokenizer_from_dict,
)
from tnsr_core.tokenizer.bpe import pretokenize

SAMPLES = [
    "Hello, world! It's 2026.",
    "  leading and trailing spaces  \n\nnew paragraphs\ttabs",
    "ünïcödé, 日本語 and emoji 🙂👍🏽",
    "snake_case and under__scores_",
    "",
]
CORPUS = ["the cat sat on the mat. " * 20, "the dog ate the bone! " * 20, *SAMPLES]


@pytest.mark.parametrize("text", SAMPLES)
def test_pretokenize_covers_all_text(text):
    assert "".join(pretokenize(text)) == text


@pytest.mark.parametrize("text", SAMPLES)
def test_bpe_roundtrip(text):
    tok = BPETokenizer.train(CORPUS, vocab_size=320)
    assert tok.decode(tok.encode(text)) == text


def test_bpe_learns_merges_and_compresses():
    tok = BPETokenizer.train(CORPUS, vocab_size=320)
    assert tok.vocab_size > 257
    text = "the cat sat on the mat."
    assert len(tok.encode(text)) < len(text.encode())


def test_bpe_training_is_deterministic():
    a = BPETokenizer.train(CORPUS, vocab_size=300)
    b = BPETokenizer.train(list(CORPUS), vocab_size=300)
    assert a.merges == b.merges and a.identity == b.identity


def test_bpe_vocab_size_minimum():
    with pytest.raises(ValueError):
        BPETokenizer.train(CORPUS, vocab_size=100)


def test_char_roundtrip_and_unknown():
    tok = CharTokenizer.train(["abc def"])
    assert tok.decode(tok.encode("bad face")) == "bad face"
    ids = tok.encode("xyz")
    assert all(i == tok.special_tokens.index("<|unk|>") for i in ids)


def test_special_tokens_are_not_parsed_from_text():
    tok = BPETokenizer.train(CORPUS, vocab_size=300)
    ids = tok.encode(f"hi {EOS}")
    assert tok.eos_id not in ids
    assert tok.encode("hi", add_eos=True)[-1] == tok.eos_id
    assert tok.decode([*tok.encode("hi"), tok.eos_id]) == "hi"
    assert tok.decode([*tok.encode("hi"), tok.eos_id], skip_special=False) == f"hi{EOS}"


def test_id_token_conversion_and_range():
    tok = CharTokenizer.train(["ab"])
    a = tok.encode("a")[0]
    assert tok.id_to_token(a) == "a"
    assert tok.id_to_token(tok.eos_id) == EOS
    with pytest.raises(ValueError):
        tok.decode([tok.vocab_size])


@pytest.mark.parametrize("cls_kind", ["bpe", "char"])
def test_serialization_roundtrip(tmp_path, cls_kind):
    tok = BPETokenizer.train(CORPUS, 300) if cls_kind == "bpe" else CharTokenizer.train(CORPUS)
    path = tmp_path / "tok.json"
    save_tokenizer(tok, path)
    loaded = load_tokenizer(path)
    assert loaded.identity == tok.identity
    assert loaded.vocab_size == tok.vocab_size
    for text in SAMPLES[:3]:
        assert loaded.encode(text) == tok.encode(text)


def test_identity_changes_with_vocab():
    a = BPETokenizer.train(CORPUS, 270)
    b = BPETokenizer.train(CORPUS, 280)
    assert a.identity != b.identity


def test_rejects_unknown_format_version():
    raw = CharTokenizer.train(["ab"]).to_dict()
    raw["format_version"] = 99
    with pytest.raises(ValueError, match="format version"):
        tokenizer_from_dict(raw)
    raw["format_version"] = 1
    raw["kind"] = "mystery"
    with pytest.raises(ValueError, match="kind"):
        tokenizer_from_dict(raw)


def test_rejects_corrupt_merges(tmp_path):
    raw = BPETokenizer.train(CORPUS, 300).to_dict()
    raw["merges"].append([5000, 5001])
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="undefined"):
        load_tokenizer(path)


def test_stream_decoder_handles_split_multibyte_characters():
    tok = BPETokenizer.train(["plain ascii only"], vocab_size=260)
    text = "日本 🙂 ok"
    ids = tok.encode(text)
    # Byte-level tokens split multi-byte characters; the stream must never emit U+FFFD.
    dec = tok.stream_decoder()
    pieces = [dec.push(i) for i in ids] + [dec.flush()]
    assert "".join(pieces) == text
    assert all("�" not in p for p in pieces)

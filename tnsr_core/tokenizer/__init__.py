"""Tokenizers: character-level and byte-level BPE, with versioned serialization."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .base import EOS, TOKENIZER_FORMAT_VERSION, UNK, StreamDecoder, Tokenizer
from .bpe import BPETokenizer
from .char import CharTokenizer

__all__ = [
    "EOS",
    "UNK",
    "BPETokenizer",
    "CharTokenizer",
    "StreamDecoder",
    "Tokenizer",
    "load_tokenizer",
    "save_tokenizer",
    "tokenizer_from_dict",
]


def tokenizer_from_dict(raw: dict[str, Any]) -> Tokenizer:
    version = raw.get("format_version")
    if version != TOKENIZER_FORMAT_VERSION:
        raise ValueError(
            f"Unsupported tokenizer format version {version!r} (expected {TOKENIZER_FORMAT_VERSION})"
        )
    kind = raw.get("kind")
    if kind == "char":
        return CharTokenizer.from_dict(raw)
    if kind == "bpe":
        return BPETokenizer.from_dict(raw)
    if kind == "chess":
        from tnsr_core.chess.encoding import ChessTokenizer

        return ChessTokenizer.from_dict(raw)
    raise ValueError(f"Unknown tokenizer kind {kind!r}")


def save_tokenizer(tokenizer: Tokenizer, path: str | Path) -> None:
    Path(path).write_text(tokenizer.to_json(), encoding="utf-8")


def load_tokenizer(path: str | Path) -> Tokenizer:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Tokenizer file not found: {path}")
    return tokenizer_from_dict(json.loads(path.read_text(encoding="utf-8")))

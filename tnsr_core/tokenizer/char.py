"""Character-level tokenizer for quick experiments."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from .base import EOS, UNK, Tokenizer


class CharTokenizer(Tokenizer):
    kind = "char"

    def __init__(self, chars: list[str]):
        if len(set(chars)) != len(chars):
            raise ValueError("Duplicate characters in vocabulary")
        if any(len(c) != 1 for c in chars):
            raise ValueError("Character vocabulary entries must be single characters")
        self.special_tokens = [EOS, UNK]
        self.chars = list(chars)
        offset = len(self.special_tokens)
        self._stoi = {c: i + offset for i, c in enumerate(self.chars)}
        self._unk = self.special_tokens.index(UNK)

    @classmethod
    def train(cls, texts: Iterable[str]) -> CharTokenizer:
        seen: set[str] = set()
        for text in texts:
            seen.update(text)
        return cls(sorted(seen))

    @property
    def vocab_size(self) -> int:
        return len(self.special_tokens) + len(self.chars)

    def encode(self, text: str, add_eos: bool = False) -> list[int]:
        ids = [self._stoi.get(c, self._unk) for c in text]
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def token_bytes(self, token_id: int) -> bytes:
        self._check_id(token_id)
        if token_id in self.special_ids:
            return b""
        return self.chars[token_id - len(self.special_tokens)].encode("utf-8")

    def _payload(self) -> dict[str, Any]:
        return {"chars": self.chars}

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> CharTokenizer:
        tok = cls(raw["chars"])
        if raw.get("special_tokens", tok.special_tokens) != tok.special_tokens:
            raise ValueError("Unsupported special token layout for char tokenizer")
        return tok

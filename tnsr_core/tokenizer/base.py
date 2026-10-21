"""Tokenizer interface, special tokens and streaming detokenization."""

from __future__ import annotations

import codecs
import hashlib
import json
from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any

EOS = "<|eos|>"
UNK = "<|unk|>"
TOKENIZER_FORMAT_VERSION = 1


class Tokenizer(ABC):
    """Maps text to integer ids and back.

    Special tokens are never parsed out of input text: a literal ``<|eos|>``
    in user text is encoded as ordinary characters. Special ids are inserted
    programmatically (``add_eos=True``) so untrusted text cannot forge them.
    """

    kind: str = "base"
    special_tokens: list[str]

    @property
    @abstractmethod
    def vocab_size(self) -> int: ...

    @property
    def eos_id(self) -> int:
        return self.special_tokens.index(EOS)

    @property
    def special_ids(self) -> set[int]:
        return set(range(len(self.special_tokens)))

    @abstractmethod
    def encode(self, text: str, add_eos: bool = False) -> list[int]: ...

    @abstractmethod
    def token_bytes(self, token_id: int) -> bytes:
        """UTF-8 bytes for a non-special token id."""

    @abstractmethod
    def _payload(self) -> dict[str, Any]: ...

    def id_to_token(self, token_id: int) -> str:
        self._check_id(token_id)
        if token_id in self.special_ids:
            return self.special_tokens[token_id]
        return self.token_bytes(token_id).decode("utf-8", errors="replace")

    def decode(self, ids: Iterable[int], skip_special: bool = True) -> str:
        out = bytearray()
        for i in ids:
            self._check_id(i)
            if i in self.special_ids:
                if not skip_special:
                    out += self.special_tokens[i].encode()
                continue
            out += self.token_bytes(i)
        return out.decode("utf-8", errors="replace")

    def _check_id(self, token_id: int) -> None:
        if not 0 <= token_id < self.vocab_size:
            raise ValueError(f"Token id {token_id} out of range for vocab size {self.vocab_size}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": TOKENIZER_FORMAT_VERSION,
            "kind": self.kind,
            "special_tokens": list(self.special_tokens),
            **self._payload(),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False)

    @property
    def identity(self) -> str:
        """Stable content hash identifying this exact vocabulary."""
        digest = hashlib.sha256(self.to_json().encode()).hexdigest()[:16]
        return f"{self.kind}-{self.vocab_size}-{digest}"

    def describe(self) -> dict[str, Any]:
        return {"kind": self.kind, "vocab_size": self.vocab_size, "identity": self.identity}

    def stream_decoder(self) -> StreamDecoder:
        return StreamDecoder(self)


class StreamDecoder:
    """Incrementally converts token ids to text without splitting UTF-8 sequences.

    Byte-level BPE tokens can end mid-character; the incremental UTF-8 decoder
    holds partial bytes back until the character is complete.
    """

    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")

    def push(self, token_id: int) -> str:
        if token_id in self.tokenizer.special_ids:
            return ""
        return self._decoder.decode(self.tokenizer.token_bytes(token_id))

    def flush(self) -> str:
        return self._decoder.decode(b"", final=True)

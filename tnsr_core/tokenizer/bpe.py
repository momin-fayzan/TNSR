"""Byte-level Byte Pair Encoding tokenizer trained from scratch.

Base vocabulary is the 256 byte values, so any UTF-8 text round-trips without
unknown tokens. Text is first split into pre-tokens (words with an optional
leading space, single digits, punctuation runs, whitespace) and merges never
cross pre-token boundaries.
"""

from __future__ import annotations

import heapq
import re
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from typing import Any

from .base import EOS, Tokenizer

# Digits are split individually so numbers tokenize consistently.
PRETOKEN_PATTERN = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?[^\W\d_]+| ?\d| ?(?:[^\s\w]|_)+|\s+(?!\S)|\s+"""
)

_CACHE_LIMIT = 50_000


def pretokenize(text: str) -> list[str]:
    return PRETOKEN_PATTERN.findall(text)


class BPETokenizer(Tokenizer):
    kind = "bpe"

    def __init__(self, merges: list[tuple[int, int]]):
        self.special_tokens = [EOS]
        self.merges = [tuple(m) for m in merges]
        self._offset = len(self.special_tokens)
        self._vocab: list[bytes] = [bytes([b]) for b in range(256)]
        self._ranks: dict[tuple[int, int], int] = {}
        for rank, (a, b) in enumerate(self.merges):
            max_id = self._offset + len(self._vocab)
            if not (self._offset <= a < max_id and self._offset <= b < max_id):
                raise ValueError(f"Merge {rank} references undefined token ids {(a, b)}")
            self._vocab.append(self._vocab[a - self._offset] + self._vocab[b - self._offset])
            self._ranks[(a, b)] = rank
        self._cache: dict[str, list[int]] = {}

    @property
    def vocab_size(self) -> int:
        return self._offset + len(self._vocab)

    def token_bytes(self, token_id: int) -> bytes:
        self._check_id(token_id)
        if token_id < self._offset:
            return b""
        return self._vocab[token_id - self._offset]

    # -- training -----------------------------------------------------------------

    @classmethod
    def train(
        cls,
        texts: Iterable[str],
        vocab_size: int,
        min_frequency: int = 2,
        progress: Callable[[int, int], None] | None = None,
    ) -> BPETokenizer:
        """Learn merges until ``vocab_size`` or no pair occurs ``min_frequency`` times."""
        base = 1 + 256
        if vocab_size < base:
            raise ValueError(f"BPE vocab_size must be >= {base} (special tokens + 256 bytes)")
        word_counts: Counter[str] = Counter()
        for text in texts:
            word_counts.update(pretokenize(text))

        words: list[list[int]] = []
        counts: list[int] = []
        for word, count in word_counts.items():
            words.append([b + 1 for b in word.encode("utf-8")])
            counts.append(count)

        pair_counts: dict[tuple[int, int], int] = defaultdict(int)
        pair_words: dict[tuple[int, int], set[int]] = defaultdict(set)
        for wi, ids in enumerate(words):
            for pair in zip(ids, ids[1:], strict=False):
                pair_counts[pair] += counts[wi]
                pair_words[pair].add(wi)

        # Lazy max-heap; ties broken by smallest pair for deterministic results.
        heap = [(-c, p) for p, c in pair_counts.items()]
        heapq.heapify(heap)
        merges: list[tuple[int, int]] = []
        target = vocab_size - base
        while len(merges) < target and heap:
            neg, pair = heapq.heappop(heap)
            current = pair_counts.get(pair, 0)
            if -neg != current:
                continue  # stale heap entry
            if current < min_frequency:
                break
            new_id = base + len(merges)
            merges.append(pair)
            changed: set[tuple[int, int]] = set()
            for wi in list(pair_words.pop(pair, ())):
                ids = words[wi]
                c = counts[wi]
                merged = _merge(ids, pair, new_id)
                if len(merged) == len(ids):
                    continue
                for p in zip(ids, ids[1:], strict=False):
                    pair_counts[p] -= c
                    changed.add(p)
                for p in zip(merged, merged[1:], strict=False):
                    pair_counts[p] += c
                    pair_words[p].add(wi)
                    changed.add(p)
                words[wi] = merged
            pair_counts.pop(pair, None)
            for p in changed:
                c = pair_counts.get(p, 0)
                if c <= 0:
                    pair_counts.pop(p, None)
                elif p != pair:
                    heapq.heappush(heap, (-c, p))
            if progress and len(merges) % 100 == 0:
                progress(len(merges), target)
        return cls(merges)

    # -- encoding -----------------------------------------------------------------

    def _encode_word(self, word: str) -> list[int]:
        cached = self._cache.get(word)
        if cached is not None:
            return cached
        ids = [b + self._offset for b in word.encode("utf-8")]
        ranks = self._ranks
        while len(ids) > 1:
            best_rank = None
            best_pair = None
            for pair in zip(ids, ids[1:], strict=False):
                r = ranks.get(pair)
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank, best_pair = r, pair
            if best_pair is None:
                break
            assert best_rank is not None
            ids = _merge(ids, best_pair, self._offset + 256 + best_rank)
        if len(self._cache) < _CACHE_LIMIT:
            self._cache[word] = ids
        return ids

    def encode(self, text: str, add_eos: bool = False) -> list[int]:
        out: list[int] = []
        for word in pretokenize(text):
            out.extend(self._encode_word(word))
        if add_eos:
            out.append(self.eos_id)
        return out

    def _payload(self) -> dict[str, Any]:
        return {"merges": [list(m) for m in self.merges]}

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> BPETokenizer:
        tok = cls([tuple(m) for m in raw["merges"]])
        if raw.get("special_tokens", tok.special_tokens) != tok.special_tokens:
            raise ValueError("Unsupported special token layout for BPE tokenizer")
        return tok


def _merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    out: list[int] = []
    i = 0
    n = len(ids)
    a, b = pair
    while i < n:
        if i < n - 1 and ids[i] == a and ids[i + 1] == b:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out

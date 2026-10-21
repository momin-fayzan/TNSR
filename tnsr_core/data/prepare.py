"""Corpus loading, reproducible document-level splits, tokenizer fitting and token files.

A prepared dataset lives in ``datasets/<name>/``:

    train.bin / val.bin   flat token streams (uint16 or uint32), EOS after each document
    tokenizer.json        tokenizer fitted on the *training split only*
    meta.json             provenance, split parameters, statistics and identity hashes
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tnsr_core.paths import datasets_dir, safe_join, validate_name
from tnsr_core.tokenizer import BPETokenizer, CharTokenizer, Tokenizer, save_tokenizer

DATASET_FORMAT_VERSION = 1
MAX_CORPUS_BYTES = 2 * 1024**3


class DatasetError(ValueError):
    pass


def read_text_documents(
    path: str | Path, mode: str = "paragraph"
) -> tuple[list[str], dict[str, Any]]:
    """Read ``.txt``/``.md`` files from a file or directory.

    ``mode='paragraph'`` splits each file on blank lines; ``mode='file'`` treats
    each file as one document; ``mode='line'`` treats each non-empty line as one.
    """
    path = Path(path).expanduser()
    if not path.exists():
        raise DatasetError(f"Corpus path not found: {path}")
    if mode not in {"paragraph", "file", "line"}:
        raise DatasetError("Split mode must be 'paragraph', 'file' or 'line'")
    files = (
        [path]
        if path.is_file()
        else sorted(
            p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in {".txt", ".md"}
        )
    )
    if not files:
        raise DatasetError(f"No .txt or .md files found under {path}")
    total = sum(f.stat().st_size for f in files)
    if total > MAX_CORPUS_BYTES:
        raise DatasetError(
            f"Corpus is {total / 1e9:.1f} GB; the in-memory pipeline supports up to 2 GB"
        )
    docs: list[str] = []
    hasher = hashlib.sha256()
    for f in files:
        raw = f.read_bytes()
        hasher.update(raw)
        text = raw.decode("utf-8", errors="replace").replace("\r\n", "\n")
        if mode == "file":
            parts = [text]
        elif mode == "line":
            parts = text.split("\n")
        else:
            parts = text.split("\n\n")
        docs.extend(p.strip() for p in parts if p.strip())
    info = {
        "source": f"files:{path.name}",
        "files": len(files),
        "split_mode": mode,
        "source_sha256": hasher.hexdigest(),
        "license": "user-supplied (verify you have the right to train on this text)",
    }
    return docs, info


def split_documents(
    docs: Iterable[str], val_fraction: float, seed: int
) -> tuple[list[str], list[str]]:
    """Assign each document to train or val by hashing its content with ``seed``.

    Identical documents always land in the same split, so exact duplicates can
    never leak from train into validation, and the split is independent of order.
    """
    if not 0 < val_fraction < 1:
        raise DatasetError("val_fraction must be in (0, 1)")
    train: list[str] = []
    val: list[str] = []
    for doc in docs:
        h = hashlib.sha256(f"{seed}:{doc}".encode()).digest()
        u = int.from_bytes(h[:8], "big") / 2**64
        (val if u < val_fraction else train).append(doc)
    if not train or not val:
        raise DatasetError(
            f"Split produced {len(train)} train / {len(val)} val documents; need at least one of each "
            "(add more documents or adjust val_fraction)"
        )
    return train, val


@dataclass
class PreparedDataset:
    name: str
    path: Path
    meta: dict[str, Any]


def _token_dtype(vocab_size: int) -> np.dtype:
    return np.dtype(np.uint16) if vocab_size <= 2**16 else np.dtype(np.uint32)


def _encode_split(tokenizer: Tokenizer, docs: list[str], dtype: np.dtype) -> np.ndarray:
    chunks = [np.asarray(tokenizer.encode(d, add_eos=True), dtype=dtype) for d in docs]
    return np.concatenate(chunks) if chunks else np.zeros(0, dtype=dtype)


def prepare_dataset(
    name: str,
    docs: list[str],
    *,
    tokenizer_kind: str = "bpe",
    vocab_size: int = 2048,
    val_fraction: float = 0.05,
    seed: int = 0,
    provenance: dict[str, Any] | None = None,
    overwrite: bool = False,
    root: Path | None = None,
    log: Callable[[str], None] = lambda _m: None,
) -> PreparedDataset:
    validate_name(name, "dataset name")
    root = root or datasets_dir()
    out = safe_join(root, name)
    if out.exists():
        if not overwrite:
            raise DatasetError(
                f"Dataset {name!r} already exists at {out} (use --overwrite to replace it)"
            )
        shutil.rmtree(out)
    if not docs:
        raise DatasetError("Corpus contains no documents")

    t0 = time.time()
    train_docs, val_docs = split_documents(docs, val_fraction, seed)
    log(f"Split {len(docs)} documents -> {len(train_docs)} train / {len(val_docs)} val")

    # The tokenizer only ever sees the training split.
    if tokenizer_kind == "char":
        tokenizer: Tokenizer = CharTokenizer.train(train_docs)
    elif tokenizer_kind == "bpe":
        log(f"Training byte-level BPE tokenizer (target vocab {vocab_size}) on the train split")
        tokenizer = BPETokenizer.train(train_docs, vocab_size)
    else:
        raise DatasetError("tokenizer must be 'char' or 'bpe'")
    dtype = _token_dtype(tokenizer.vocab_size)

    train_ids = _encode_split(tokenizer, train_docs, dtype)
    val_ids = _encode_split(tokenizer, val_docs, dtype)

    tmp = out.with_name(out.name + ".partial")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    train_ids.tofile(tmp / "train.bin")
    val_ids.tofile(tmp / "val.bin")
    save_tokenizer(tokenizer, tmp / "tokenizer.json")

    content_hash = hashlib.sha256()
    content_hash.update(train_ids.tobytes())
    content_hash.update(b"|")
    content_hash.update(val_ids.tobytes())
    val_text_hash = hashlib.sha256("\x00".join(val_docs).encode()).hexdigest()

    train_bytes = sum(len(d.encode()) for d in train_docs)
    val_bytes = sum(len(d.encode()) for d in val_docs)
    meta = {
        "format_version": DATASET_FORMAT_VERSION,
        "name": name,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "identity": f"{name}-{content_hash.hexdigest()[:12]}",
        "val_text_sha256": val_text_hash,
        "provenance": provenance or {},
        "split": {"method": "sha256(seed:document)", "seed": seed, "val_fraction": val_fraction},
        "tokenizer": tokenizer.describe(),
        "dtype": dtype.name,
        "stats": {
            "documents": len(docs),
            "train_documents": len(train_docs),
            "val_documents": len(val_docs),
            "train_tokens": int(train_ids.size),
            "val_tokens": int(val_ids.size),
            "train_bytes": train_bytes,
            "val_bytes": val_bytes,
            "train_chars": sum(len(d) for d in train_docs),
            "val_chars": sum(len(d) for d in val_docs),
            "bytes_per_token": round(train_bytes / max(1, train_ids.size), 4),
            "mean_doc_chars": round(sum(len(d) for d in docs) / len(docs), 1),
        },
        "prepare_seconds": round(time.time() - t0, 2),
    }
    (tmp / "meta.json").write_text(json.dumps(meta, indent=2))
    tmp.rename(out)
    log(f"Wrote dataset to {out}")
    return PreparedDataset(name, out, meta)

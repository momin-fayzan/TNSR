"""Typed, validated configuration for models, data and training runs."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import tomllib

PRESETS_DIR = Path(__file__).parent / "presets"


class ConfigError(ValueError):
    """Raised for invalid or inconsistent configuration values."""


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise ConfigError(msg)


@dataclass
class ModelConfig:
    vocab_size: int = 256
    block_size: int = 128
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.0
    bias: bool = False
    norm: str = "rmsnorm"  # "rmsnorm" | "layernorm"
    pos_encoding: str = "learned"  # "learned" | "rope"
    mlp_ratio: float = 4.0
    tie_embeddings: bool = True
    attn_impl: str = "sdpa"  # "sdpa" (fused PyTorch kernel) | "manual" (explicit mask)
    causal: bool = True  # False = bidirectional attention (encoder, e.g. chess positions)
    num_classes: int = 0  # 0 = next-token LM head; >0 = classify from the last position

    def validate(self) -> ModelConfig:
        _check(self.vocab_size >= 2, "model.vocab_size must be >= 2")
        _check(self.block_size >= 2, "model.block_size must be >= 2")
        _check(self.n_layer >= 1, "model.n_layer must be >= 1")
        _check(self.n_head >= 1, "model.n_head must be >= 1")
        _check(self.n_embd >= 1, "model.n_embd must be >= 1")
        _check(
            self.n_embd % self.n_head == 0,
            f"model.n_embd ({self.n_embd}) must be divisible by model.n_head ({self.n_head})",
        )
        _check(0.0 <= self.dropout < 1.0, "model.dropout must be in [0, 1)")
        _check(self.norm in {"rmsnorm", "layernorm"}, "model.norm must be 'rmsnorm' or 'layernorm'")
        _check(
            self.pos_encoding in {"learned", "rope"},
            "model.pos_encoding must be 'learned' or 'rope'",
        )
        if self.pos_encoding == "rope":
            _check((self.n_embd // self.n_head) % 2 == 0, "RoPE requires an even head dimension")
        _check(self.mlp_ratio > 0, "model.mlp_ratio must be > 0")
        _check(self.attn_impl in {"sdpa", "manual"}, "model.attn_impl must be 'sdpa' or 'manual'")
        _check(
            self.num_classes == 0 or self.num_classes >= 2,
            "model.num_classes must be 0 (language model) or >= 2 (classifier)",
        )
        return self

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head


@dataclass
class DataConfig:
    dataset: str = "demo"  # name of a prepared dataset under datasets/

    def validate(self) -> DataConfig:
        _check(bool(self.dataset), "data.dataset must be set")
        return self


@dataclass
class TrainConfig:
    seed: int = 1337
    device: str = "auto"
    precision: str = "auto"  # auto | fp32 | bf16 | fp16
    batch_size: int = 16
    grad_accum_steps: int = 1
    max_steps: int = 1000
    learning_rate: float = 3e-3
    min_lr: float = 3e-4
    warmup_steps: int = 50
    lr_schedule: str = "cosine"  # cosine | linear | constant
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    eval_interval: int = 100
    eval_iters: int = 20
    log_interval: int = 10
    checkpoint_interval: int = 0  # 0 = checkpoint at every evaluation
    keep_checkpoints: int = 3
    compile: bool = False
    max_nonfinite: int = 3  # consecutive non-finite losses before aborting
    num_threads: int = 0  # 0 = PyTorch default

    def validate(self) -> TrainConfig:
        _check(self.batch_size >= 1, "train.batch_size must be >= 1")
        _check(self.grad_accum_steps >= 1, "train.grad_accum_steps must be >= 1")
        _check(self.max_steps >= 1, "train.max_steps must be >= 1")
        _check(self.learning_rate > 0, "train.learning_rate must be > 0")
        _check(0 <= self.min_lr <= self.learning_rate, "train.min_lr must be in [0, learning_rate]")
        _check(self.warmup_steps >= 0, "train.warmup_steps must be >= 0")
        _check(
            self.lr_schedule in {"cosine", "linear", "constant"},
            "train.lr_schedule must be cosine, linear or constant",
        )
        _check(self.weight_decay >= 0, "train.weight_decay must be >= 0")
        _check(0 < self.beta1 < 1 and 0 < self.beta2 < 1, "train betas must be in (0, 1)")
        _check(self.grad_clip >= 0, "train.grad_clip must be >= 0 (0 disables clipping)")
        _check(self.eval_interval >= 1, "train.eval_interval must be >= 1")
        _check(self.eval_iters >= 1, "train.eval_iters must be >= 1")
        _check(self.log_interval >= 1, "train.log_interval must be >= 1")
        _check(self.checkpoint_interval >= 0, "train.checkpoint_interval must be >= 0")
        _check(self.keep_checkpoints >= 1, "train.keep_checkpoints must be >= 1")
        _check(
            self.precision in {"auto", "fp32", "bf16", "fp16"},
            "train.precision must be auto, fp32, bf16 or fp16",
        )
        _check(
            self.device in {"auto", "cpu", "cuda", "mps"},
            "train.device must be auto, cpu, cuda or mps",
        )
        _check(self.max_nonfinite >= 1, "train.max_nonfinite must be >= 1")
        _check(self.num_threads >= 0, "train.num_threads must be >= 0")
        return self


@dataclass
class RunConfig:
    name: str = "run"
    description: str = ""
    task: str = "lm"  # "lm" (language modeling) | "chess" (action-value chess engine)
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def validate(self) -> RunConfig:
        _check(self.task in {"lm", "chess"}, "task must be 'lm' or 'chess'")
        self.model.validate()
        self.data.validate()
        self.train.validate()
        return self

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> RunConfig:
        raw = dict(raw)
        unknown = set(raw) - {"name", "description", "task", "model", "data", "train"}
        _check(not unknown, f"Unknown top-level config keys: {sorted(unknown)}")
        return cls(
            name=raw.get("name", "run"),
            description=raw.get("description", ""),
            task=raw.get("task", "lm"),
            model=_build(ModelConfig, raw.get("model", {}), "model"),
            data=_build(DataConfig, raw.get("data", {}), "data"),
            train=_build(TrainConfig, raw.get("train", {}), "train"),
        ).validate()


def _build(cls: type, values: dict[str, Any], section: str) -> Any:
    names = {f.name: f for f in dataclasses.fields(cls)}
    unknown = set(values) - set(names)
    _check(not unknown, f"Unknown keys in [{section}]: {sorted(unknown)}")
    kwargs = {}
    for key, value in values.items():
        expected = type(getattr(cls(), key))
        if expected is float and isinstance(value, int) and not isinstance(value, bool):
            value = float(value)
        _check(
            isinstance(value, expected) and not (expected is int and isinstance(value, bool)),
            f"[{section}].{key} must be {expected.__name__}, got {type(value).__name__}",
        )
        kwargs[key] = value
    return cls(**kwargs)


def model_config_from_dict(raw: dict[str, Any]) -> ModelConfig:
    return _build(ModelConfig, raw, "model").validate()


def list_presets() -> list[str]:
    return sorted(p.stem for p in PRESETS_DIR.glob("*.toml"))


def load_config(name_or_path: str, overrides: dict[str, Any] | None = None) -> RunConfig:
    """Load a preset name (e.g. ``tiny``) or a TOML file path, applying dotted overrides."""
    path = Path(name_or_path)
    if not path.suffix:
        path = PRESETS_DIR / f"{name_or_path}.toml"
    if not path.is_file():
        raise ConfigError(
            f"Config {name_or_path!r} not found. Presets: {', '.join(list_presets())}"
        )
    with path.open("rb") as fh:
        raw = tomllib.load(fh)
    for dotted, value in (overrides or {}).items():
        section, _, key = dotted.partition(".")
        if not key:
            raw[section] = value
        else:
            raw.setdefault(section, {})[key] = value
    return RunConfig.from_dict(raw)


def parse_override(text: str) -> tuple[str, Any]:
    """Parse ``section.key=value`` with TOML value semantics (strings may be bare)."""
    if "=" not in text:
        raise ConfigError(f"Override {text!r} must look like section.key=value")
    key, _, value = text.partition("=")
    key = key.strip()
    try:
        parsed = tomllib.loads(f"v = {value.strip()}")["v"]
    except tomllib.TOMLDecodeError:
        parsed = value.strip()
    return key, parsed

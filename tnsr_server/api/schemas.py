"""Request/response schemas (validated with pydantic)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from tnsr_core.inference import GenerationConfig
from tnsr_core.inference.sampling import MAX_NEW_TOKENS_LIMIT


class GenerationSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_new_tokens: int = Field(160, ge=1, le=MAX_NEW_TOKENS_LIMIT)
    temperature: float = Field(0.7, ge=0.0, le=5.0)
    top_k: int = Field(40, ge=0, le=65536)
    top_p: float = Field(0.95, gt=0.0, le=1.0)
    repetition_penalty: float = Field(1.0, ge=1.0, le=3.0)
    deterministic: bool = False
    seed: int | None = Field(None, ge=0, lt=2**53)
    ignore_eos: bool = False

    def to_config(self, stop: list[str] | None = None) -> GenerationConfig:
        return GenerationConfig(
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            repetition_penalty=self.repetition_penalty,
            deterministic=self.deterministic,
            seed=self.seed,
            stop=list(stop or []),
            stop_at_eos=not self.ignore_eos,
        )


Mode = Literal["chat", "completion"]


class ConversationCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str | None = Field(None, max_length=200)
    model_id: str | None = Field(None, max_length=512)
    mode: Mode = "chat"
    settings: GenerationSettings | None = None


class ConversationUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str | None = Field(None, min_length=1, max_length=200)
    model_id: str | None = Field(None, max_length=512)
    mode: Mode | None = None
    settings: GenerationSettings | None = None

    @field_validator("title")
    @classmethod
    def strip_title(cls, v: str | None) -> str | None:
        if v is not None:
            v = v.strip()
            if not v:
                raise ValueError("title must not be blank")
        return v


class MessageCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    content: str = Field(..., min_length=1, max_length=8000)
    settings: GenerationSettings | None = None
    model_id: str | None = Field(None, max_length=512)

    @field_validator("content")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("content must not be blank")
        return v


class RetryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    settings: GenerationSettings | None = None


class CompletionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(..., max_length=20000)
    model_id: str | None = Field(None, max_length=512)
    settings: GenerationSettings = Field(default_factory=GenerationSettings)
    stop: list[str] = Field(default_factory=list, max_length=8)


class LoadModelRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_id: str = Field(..., min_length=1, max_length=512)


class AppSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    theme: Literal["dark", "light", "system"] = "dark"
    generation: GenerationSettings = Field(default_factory=GenerationSettings)
    default_model_id: str | None = None
    default_mode: Mode = "chat"


class JobCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["train", "resume", "evaluate", "benchmark", "prepare"]
    params: dict[str, Any] = Field(default_factory=dict)


class CompareRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_ids: list[str] = Field(..., min_length=1, max_length=8)


class GameCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    human_color: Literal["white", "black"] = "white"
    model_id: str | None = Field(None, max_length=512)
    start_fen: str | None = Field(None, max_length=100)


class GameUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(..., min_length=1, max_length=200)


class MoveRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    move: str = Field(..., pattern=r"^[a-h][1-8][a-h][1-8][qrbn]?$")


class AnalyseRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fen: str = Field(..., max_length=100)
    model_id: str | None = Field(None, max_length=512)

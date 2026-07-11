from __future__ import annotations

from pydantic import BaseModel, Field


class ModelSettings(BaseModel):
    temperature: float = 0.0
    max_tokens: int = 2048
    timeout_seconds: float = 120.0
    extra: dict[str, object] = Field(default_factory=dict)

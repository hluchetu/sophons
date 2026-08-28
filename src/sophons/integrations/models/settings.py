from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class ModelSettings:
    temperature: float = 0.0
    max_tokens: int = 2048
    timeout_seconds: float = 120.0
    extra: dict[str, object] = field(default_factory=dict)

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Literal


MessageRole = Literal["system", "user", "assistant", "tool"]


@dataclass(frozen=True, slots=True)
class Message:
    """Provider-neutral chat message used by Sophons model adapters."""

    role: MessageRole
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str | None = None

    def with_metadata(self, **metadata: Any) -> Message:
        """Return a copy with additional metadata."""

        return replace(self, metadata={**self.metadata, **metadata})

    def with_id(self, id: str | None) -> Message:
        """Return a copy with a message ID."""

        return replace(self, id=id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "role": self.role,
            "content": self.content,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Message:
        return cls(
            id=data.get("id"),
            role=data["role"],
            content=data["content"],
            metadata=dict(data.get("metadata") or {}),
        )

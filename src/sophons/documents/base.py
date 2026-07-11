from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any


@dataclass(frozen=True, slots=True)
class Document:
    """Provider-neutral text document used across Sophons."""

    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str | None = None
    score: float | None = None

    def with_metadata(self, **metadata: Any) -> Document:
        """Return a copy with additional metadata."""

        return replace(self, metadata={**self.metadata, **metadata})

    def with_score(self, score: float | None) -> Document:
        """Return a copy with a retrieval score."""

        return replace(self, score=score)

    def with_id(self, id: str | None) -> Document:
        """Return a copy with a document ID."""

        return replace(self, id=id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "metadata": dict(self.metadata),
            "score": self.score,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Document:
        return cls(
            id=data.get("id"),
            content=data["content"],
            metadata=dict(data.get("metadata") or {}),
            score=data.get("score"),
        )

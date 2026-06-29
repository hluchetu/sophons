from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4


# ---------------------------------------------------------------------------
# MemoryType
# ---------------------------------------------------------------------------

MemoryType = Literal[
    "semantic",    # general facts and knowledge  — "Python is a programming language"
    "entity",      # named things and their attributes — "John's email is john@example.com"
    "episodic",    # specific past events — "User asked about async on Monday"
    "procedural",  # rules and workflows — "Always check docs before answering"
    "preference",  # user preferences — "User prefers concise answers"
    "decision",    # choices made — "User chose option A over option B"
]

ALL_MEMORY_TYPES: tuple[MemoryType, ...] = (
    "semantic",
    "entity",
    "episodic",
    "procedural",
    "preference",
    "decision",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _new_id() -> str:
    return str(uuid4())


def _utc_now() -> datetime:
    return datetime.now(tz=timezone.utc)


# ---------------------------------------------------------------------------
# MemoryEntry
# ---------------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class MemoryEntry:
    """
    A single unit of long-term memory.

    Every fact, preference, event, or rule the agent learns is stored as a
    ``MemoryEntry``.  Entries are identified by their ``namespace`` + ``key``
    pair and retrieved by type, recency, importance, or semantic similarity.

    Attributes:
        memory_type:  What kind of memory this is.  Controls which retrieval
                      strategies are applied.
        namespace:    Hierarchical owner path — e.g. ``("user", "alice")`` or
                      ``("project", "sophons")``.  Acts like a folder path so
                      memories can be scoped and access-controlled.
        key:          Unique name within the namespace.  Together with
                      ``namespace`` it uniquely identifies this entry in the
                      store.
        content:      The actual fact, preference, event, or rule as a plain
                      string.
        id:           Auto-generated UUID.  Stable across updates.
        created_at:   UTC timestamp of when the entry was first stored.
        expires_at:   Optional UTC timestamp after which the entry is
                      considered stale.  ``None`` means it never expires.
        invalidated_at: Set when a newer entry supersedes this one.  The old
                      entry is kept for history but excluded from retrieval.
        importance:   Optional float in [0, 1] indicating how significant this
                      memory is.  Higher importance entries are ranked first
                      when results are trimmed.
        related_ids:  IDs of other entries that are semantically or causally
                      linked to this one.
        metadata:     Arbitrary key-value data — source document ID, model
                      that extracted it, confidence score, etc.
    """

    memory_type: MemoryType
    namespace: tuple[str, ...]
    key: str
    content: str
    id: str = field(default_factory=_new_id)
    created_at: datetime = field(default_factory=_utc_now)
    expires_at: datetime | None = None
    invalidated_at: datetime | None = None
    importance: float | None = None
    related_ids: tuple[str, ...] = field(default=())
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.namespace:
            raise ValueError("namespace must not be empty.")
        if not self.key.strip():
            raise ValueError("key must not be empty or whitespace.")
        if not self.content.strip():
            raise ValueError("content must not be empty or whitespace.")
        if self.importance is not None and not (0.0 <= self.importance <= 1.0):
            raise ValueError("importance must be between 0.0 and 1.0.")

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    @property
    def is_valid(self) -> bool:
        """True if the entry has not been invalidated."""
        return self.invalidated_at is None

    @property
    def is_expired(self) -> bool:
        """True if the entry has passed its expiry time."""
        if self.expires_at is None:
            return False
        return _utc_now() >= self.expires_at

    @property
    def is_active(self) -> bool:
        """True if the entry is valid and not expired."""
        return self.is_valid and not self.is_expired

    def namespace_str(self) -> str:
        """Return the namespace as a slash-separated string."""
        return "/".join(self.namespace)

    # ------------------------------------------------------------------
    # Mutation helpers (return new instances — entries are frozen)
    # ------------------------------------------------------------------

    def invalidate(self) -> MemoryEntry:
        """Return a copy marked as invalidated at the current UTC time."""
        from dataclasses import replace
        return replace(self, invalidated_at=_utc_now())

    def with_importance(self, importance: float) -> MemoryEntry:
        """Return a copy with a new importance score."""
        from dataclasses import replace
        return replace(self, importance=importance)

    def with_related(self, *ids: str) -> MemoryEntry:
        """Return a copy with additional related IDs appended."""
        from dataclasses import replace
        return replace(self, related_ids=self.related_ids + ids)

    def with_metadata(self, **kwargs: Any) -> MemoryEntry:
        """Return a copy with additional metadata merged in."""
        from dataclasses import replace
        return replace(self, metadata={**self.metadata, **kwargs})

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "memory_type": self.memory_type,
            "namespace": list(self.namespace),
            "key": self.key,
            "content": self.content,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "invalidated_at": self.invalidated_at.isoformat() if self.invalidated_at else None,
            "importance": self.importance,
            "related_ids": list(self.related_ids),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MemoryEntry:
        def _parse_dt(value: str | None) -> datetime | None:
            return datetime.fromisoformat(value) if value else None

        return cls(
            id=data["id"],
            memory_type=data["memory_type"],
            namespace=tuple(data["namespace"]),
            key=data["key"],
            content=data["content"],
            created_at=datetime.fromisoformat(data["created_at"]),
            expires_at=_parse_dt(data.get("expires_at")),
            invalidated_at=_parse_dt(data.get("invalidated_at")),
            importance=data.get("importance"),
            related_ids=tuple(data.get("related_ids", [])),
            metadata=dict(data.get("metadata", {})),
        )

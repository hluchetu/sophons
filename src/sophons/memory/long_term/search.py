from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from sophons.memory.long_term.entry import MemoryEntry, MemoryType


# ---------------------------------------------------------------------------
# MetadataFilter
# ---------------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class MetadataFilter:
    """
    Filters applied to entry metadata during search.

    All conditions are AND-ed together — an entry must satisfy every
    condition that is set to be included in results.

    Attributes:
        equals:            Entries whose metadata contains all these key-value
                           pairs pass. Example: ``{"source": "chat"}``.
        contains_all_tags: Entries whose metadata ``tags`` list contains all
                           of these tags pass.
        created_after:     Only entries created after this UTC datetime pass.
        created_before:    Only entries created before this UTC datetime pass.
    """

    equals: dict[str, Any] = field(default_factory=dict)
    contains_all_tags: set[str] = field(default_factory=set)
    created_after: datetime | None = None
    created_before: datetime | None = None

    def matches(self, entry: MemoryEntry) -> bool:
        """Return True if ``entry`` satisfies all filter conditions."""
        for key, expected in self.equals.items():
            if entry.metadata.get(key) != expected:
                return False

        if self.contains_all_tags:
            raw = entry.metadata.get("tags", [])
            tags = set(raw) if isinstance(raw, (list, set, tuple)) else set()
            if not self.contains_all_tags.issubset(tags):
                return False

        if self.created_after is not None and entry.created_at < self.created_after:
            return False

        if self.created_before is not None and entry.created_at > self.created_before:
            return False

        return True


# ---------------------------------------------------------------------------
# MemorySearch
# ---------------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class MemorySearch:
    """
    Parameters for a memory retrieval query.

    Attributes:
        namespace:      Which namespace to search within.
        query:          Natural language question or keyword string.
        memory_type:    If set, restrict results to this type only.
        limit:          Maximum number of results to return.
        min_importance: Exclude entries with importance below this threshold.
        metadata:       Additional metadata filters applied after retrieval.
    """

    namespace: tuple[str, ...]
    query: str
    memory_type: MemoryType | None = None
    limit: int = 5
    min_importance: float | None = None
    metadata: MetadataFilter = field(default_factory=MetadataFilter)


# ---------------------------------------------------------------------------
# RetrievalResult
# ---------------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class RetrievalResult:
    """
    A single result returned by a memory retriever.

    Attributes:
        entry_id:         ID of the matching ``MemoryEntry``.
        source:           Name of the retriever that produced this result
                          e.g. ``"semantic"``, ``"lexical"``.
        score:            Final combined score used for ranking. Higher is
                          better.
        relevance_score:  Raw similarity or keyword match score before
                          re-ranking.
        recency_score:    Score component based on how recent the entry is.
        importance_score: Score component based on entry importance.
        reason:           Human-readable explanation of why this entry was
                          retrieved.
    """

    entry_id: str
    source: str
    score: float
    relevance_score: float | None = None
    recency_score: float | None = None
    importance_score: float | None = None
    reason: str | None = None

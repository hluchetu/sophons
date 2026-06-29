from __future__ import annotations

import math
from datetime import datetime, timezone

from sophons.memory.long_term.entry import MemoryEntry
from sophons.memory.long_term.search import MemorySearch, RetrievalResult
from sophons.memory.long_term.text import searchable_text
from sophons.memory.retrieval._matching import (
    blend_importance,
    entry_matches_search,
    token_overlap_score,
)


class EpisodicRetriever:
    """
    Time-aware retriever tuned for episodic memories.

    Scores entries using a blend of:

    - **Relevance** — token overlap between ``search.query`` and
      ``searchable_text(entry)``.  No embedding is needed.
    - **Recency** — exponential decay over the entry's ``created_at``
      timestamp.  Recent entries score higher.

    The final score is the arithmetic mean of the two components, then
    blended with the entry's ``importance`` field via ``blend_importance()``.

    When ``search.memory_type`` is ``None`` this retriever automatically
    scopes itself to ``"episodic"`` entries so it does not compete with
    semantic or lexical retrievers for other memory types.
    """

    #: Decay constant: score halves every ~69 hours (≈ 3 days).
    DECAY_RATE: float = 0.01

    def __init__(self) -> None:
        self._entries: dict[str, MemoryEntry] = {}

    # ------------------------------------------------------------------
    # MemoryRetriever interface
    # ------------------------------------------------------------------

    def add(self, entry: MemoryEntry) -> None:
        self._entries[entry.id] = entry

    def search(self, search: MemorySearch) -> list[RetrievalResult]:
        # Default to "episodic" when no type filter is specified
        effective_type = search.memory_type or "episodic"
        scoped = MemorySearch(
            namespace=search.namespace,
            query=search.query,
            memory_type=effective_type,
            limit=search.limit,
            min_importance=search.min_importance,
            metadata=search.metadata,
        )

        scored: list[tuple[float, float, float, MemoryEntry]] = []
        for entry in self._entries.values():
            if not entry_matches_search(entry, scoped):
                continue

            relevance = token_overlap_score(search.query, [searchable_text(entry)])
            recency = _recency_score(entry.created_at)
            combined = (relevance + recency) / 2
            scored.append((combined, relevance, recency, entry))

        scored.sort(key=lambda t: t[0], reverse=True)

        return [
            RetrievalResult(
                entry_id=entry.id,
                source="episodic",
                score=blend_importance(combined, entry),
                relevance_score=relevance,
                recency_score=recency,
                importance_score=entry.importance,
                reason="token overlap + recency decay for episodic memory",
            )
            for combined, relevance, recency, entry in scored[: scoped.limit]
        ]

    def delete(self, entry_id: str) -> None:
        self._entries.pop(entry_id, None)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _recency_score(created_at: datetime) -> float:
    """
    Exponential decay: ``exp(-DECAY_RATE * age_in_hours)``.

    Returns 1.0 for a brand-new entry, approaching 0.0 as the entry ages.
    """
    now = datetime.now(timezone.utc).timestamp()
    age_hours = max((now - created_at.timestamp()) / 3600, 0)
    return math.exp(-EpisodicRetriever.DECAY_RATE * age_hours)

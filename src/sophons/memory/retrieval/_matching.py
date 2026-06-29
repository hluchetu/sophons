from __future__ import annotations

import re
from collections.abc import Iterable
from datetime import datetime, timezone

from sophons.memory.long_term.entry import MemoryEntry, MemoryType
from sophons.memory.long_term.search import MemorySearch
from sophons.memory.long_term.text import searchable_text


# ---------------------------------------------------------------------------
# Namespace / filter helpers
# ---------------------------------------------------------------------------


def namespace_matches(
    entry_namespace: tuple[str, ...],
    search_namespace: tuple[str, ...],
) -> bool:
    """Return True if ``entry_namespace`` starts with ``search_namespace``."""
    return entry_namespace[: len(search_namespace)] == search_namespace


def memory_type_matches(
    entry: MemoryEntry,
    memory_type: MemoryType | None,
) -> bool:
    return memory_type is None or entry.memory_type == memory_type


def entry_matches_search(entry: MemoryEntry, search: MemorySearch) -> bool:
    """Return True if ``entry`` should be considered for ``search``."""
    return (
        entry_is_active(entry)
        and namespace_matches(entry.namespace, search.namespace)
        and memory_type_matches(entry, search.memory_type)
        and search.metadata.matches(entry)
    )


def entry_is_active(
    entry: MemoryEntry,
    current_time: datetime | None = None,
) -> bool:
    now = current_time or datetime.now(timezone.utc)
    if entry.invalidated_at is not None:
        return False
    return entry.expires_at is None or entry.expires_at > now


# ---------------------------------------------------------------------------
# Score helpers
# ---------------------------------------------------------------------------

IMPORTANCE_WEIGHT = 0.3


def blend_importance(score: float, entry: MemoryEntry) -> float:
    """
    Mix a raw relevance score with the entry's importance score.

    The final score is ``score * 0.7 + importance * 0.3`` when importance is
    set, otherwise returns the clamped relevance score unchanged.
    """
    relevance = clamp_score(score)
    if entry.importance is None:
        return relevance
    importance = clamp_score(entry.importance)
    return clamp_score(relevance * (1 - IMPORTANCE_WEIGHT) + importance * IMPORTANCE_WEIGHT)


def clamp_score(score: float) -> float:
    return max(0.0, min(1.0, score))


# ---------------------------------------------------------------------------
# Tokenisation helpers
# ---------------------------------------------------------------------------


def tokenize_terms(text: str) -> list[str]:
    """Lowercase word tokens — suitable for BM25 corpus and query lists."""
    return re.findall(r"[a-z0-9]+", text.lower())


def tokenize(text: str) -> set[str]:
    """Unique lowercase tokens."""
    return set(tokenize_terms(text))


def token_overlap_score(query: str, candidates: Iterable[str]) -> float:
    """
    Fraction of query tokens that appear in the union of candidate tokens.

    Returns 0.0 when the query or all candidates are empty.
    """
    query_tokens = tokenize(query)
    if not query_tokens:
        return 0.0

    candidate_tokens: set[str] = set()
    for candidate in candidates:
        candidate_tokens.update(tokenize(candidate))

    if not candidate_tokens:
        return 0.0

    overlap = query_tokens.intersection(candidate_tokens)
    return len(overlap) / len(query_tokens)

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

from sophons.memory.long_term.entry import MemoryEntry
from sophons.models.messages import Message

if TYPE_CHECKING:
    from sophons.memory.long_term.store import MemoryStore


# ---------------------------------------------------------------------------
# Request / Result
# ---------------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class MemoryExtractionRequest:
    """
    Input to a ``MemoryExtractor``.

    Args:
        namespace:        Target namespace for the extracted entries.
        messages:         Conversation to extract from (most recent last).
        existing_memories: Already-stored entries in this namespace, provided
                           so the extractor can avoid duplicates.  Pass an
                           empty list when no context is needed.
        memory_store:     Optional live store.  When provided the extractor
                          may query it for additional context.
    """

    namespace: tuple[str, ...]
    messages: list[Message]
    existing_memories: list[MemoryEntry] = field(default_factory=list)
    memory_store: MemoryStore | None = None


@dataclass(frozen=True, kw_only=True)
class MemoryExtractionResult:
    """
    Output of a ``MemoryExtractor``.

    Entries are returned to the caller rather than stored silently — the
    caller (usually ``MemoryManager``) decides where and how to persist them.
    This matches the Mem0 convention of returning explicit results from
    ``Memory.add()``.
    """

    entries: list[MemoryEntry]
    invalidated_keys: list[str] = field(default_factory=list)
    skipped_reason: str | None = None


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


class MemoryExtractor(Protocol):
    """
    Contract for a memory extraction backend.

    Implementations receive a conversation and return the ``MemoryEntry``
    objects that should be stored.  The canonical implementation is
    ``LLMMemoryExtractor``; test doubles can implement this Protocol
    directly.
    """

    async def extract(
        self, request: MemoryExtractionRequest
    ) -> MemoryExtractionResult:
        """Extract memory entries from ``request.messages``."""
        ...

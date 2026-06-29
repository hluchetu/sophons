from __future__ import annotations

import logging
from dataclasses import replace
from typing import Protocol

from sophons.memory.long_term.conflicts import find_conflicting_entries
from sophons.memory.long_term.entry import MemoryEntry, MemoryType
from sophons.memory.long_term.policy import AllowAllPolicy, NamespacePolicy
from sophons.memory.long_term.ranking import fuse_results, reciprocal_rank_score
from sophons.memory.long_term.search import MemorySearch, MetadataFilter, RetrievalResult
from sophons.memory.long_term.storage import MemoryStorage
from sophons.memory.long_term.text import searchable_text

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MemoryRetriever Protocol
# ---------------------------------------------------------------------------


class MemoryRetriever(Protocol):
    """
    Contract for a memory retrieval backend.

    A retriever indexes entries and answers search queries with ranked
    results.  Multiple retrievers can be registered on a ``MemoryStore``
    and their results are fused via Reciprocal Rank Fusion.
    """

    def add(self, entry: MemoryEntry) -> None:
        """Index a new entry."""
        ...

    def search(self, search: MemorySearch) -> list[RetrievalResult]:
        """Return ranked results for ``search``."""
        ...

    def delete(self, entry_id: str) -> None:
        """Remove an entry from the index."""
        ...


# ---------------------------------------------------------------------------
# MemoryStore
# ---------------------------------------------------------------------------


class NamespaceAccessError(Exception):
    """Raised when a namespace policy denies read or write access."""


class MemoryStore:
    """
    Orchestrates long-term memory storage and retrieval.

    ``MemoryStore`` sits on top of a ``MemoryStorage`` backend and zero or
    more ``MemoryRetriever`` instances. It handles:

    - Namespace access control via ``NamespacePolicy``
    - Conflict detection and automatic invalidation of stale entries
    - Related entry linking (entries that are semantically close are linked)
    - Ranked search via Reciprocal Rank Fusion across multiple retrievers

    Usage::

        store = MemoryStore(storage=InMemoryStorage())
        store.put(MemoryEntry(memory_type="preference", namespace=("user", "alice"),
                              key="style", content="concise"))
        results = store.search(MemorySearch(namespace=("user", "alice"),
                                            query="how does alice like answers"))
    """

    def __init__(
        self,
        storage: MemoryStorage,
        retrievers: list[MemoryRetriever] | None = None,
        namespace_policy: NamespacePolicy | None = None,
        max_related_ids: int = 3,
    ) -> None:
        self._storage = storage
        self._retrievers = retrievers or []
        self._policy = namespace_policy or AllowAllPolicy()
        self._max_related_ids = max_related_ids

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def put(self, entry: MemoryEntry) -> None:
        """
        Store an entry.

        Before storing:
        - Checks write access via the namespace policy.
        - Invalidates any existing entries that conflict with this one.
        - Populates ``related_ids`` by searching for similar entries.
        - Adds reverse links on related entries.
        """
        self._enforce_write(entry.namespace)
        self._invalidate_conflicts(entry)
        entry = self._with_related_ids(entry)
        self._storage.put(entry)
        self._add_reverse_links(entry)

        for retriever in self._retrievers:
            retriever.add(entry)

        logger.debug("put namespace=%s key=%s", entry.namespace_str(), entry.key)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get(
        self,
        namespace: tuple[str, ...],
        key: str,
    ) -> MemoryEntry | None:
        """Return the entry for ``namespace`` + ``key``, or ``None``."""
        self._enforce_read(namespace)
        return self._storage.get(namespace, key)

    def get_by_id(self, entry_id: str) -> MemoryEntry | None:
        """Return the entry with the given ID, or ``None``."""
        return self._storage.get_by_id(entry_id)

    def list(
        self,
        namespace: tuple[str, ...],
        memory_type: MemoryType | None = None,
        include_invalidated: bool = False,
    ) -> list[MemoryEntry]:
        """Return all entries in ``namespace``."""
        self._enforce_read(namespace)
        return self._storage.list(
            namespace=namespace,
            memory_type=memory_type,
            include_invalidated=include_invalidated,
        )

    def search(self, query: MemorySearch) -> list[MemoryEntry]:
        """
        Search for entries matching ``query``.

        If retrievers are registered their results are fused via Reciprocal
        Rank Fusion.  If no retrievers are registered, falls back to listing
        all active entries in the namespace filtered by type and metadata.
        """
        self._enforce_read(query.namespace)

        if not self._retrievers:
            return self._fallback_search(query)

        result_lists: list[list[RetrievalResult]] = []
        for retriever in self._retrievers:
            results = retriever.search(query)
            result_lists.append(results)

        fused = fuse_results(result_lists)[: query.limit]
        entries = self._storage.get_many([r.entry_id for r in fused])

        if query.min_importance is not None:
            entries = [
                e for e in entries
                if e.importance is None or e.importance >= query.min_importance
            ]

        if query.metadata:
            entries = [e for e in entries if query.metadata.matches(e)]

        return entries

    # ------------------------------------------------------------------
    # Invalidate / delete
    # ------------------------------------------------------------------

    def invalidate(
        self,
        namespace: tuple[str, ...],
        key: str,
    ) -> bool:
        """
        Mark an entry as invalidated without removing it from storage.

        Returns ``True`` if the entry existed and was active, ``False``
        if it did not exist or was already invalidated.
        """
        self._enforce_write(namespace)
        entry = self._storage.get(namespace, key)

        if entry is None or entry.invalidated_at is not None:
            return False

        self._storage.put(entry.invalidate())

        for retriever in self._retrievers:
            retriever.delete(entry.id)

        return True

    def delete(
        self,
        namespace: tuple[str, ...],
        key: str,
    ) -> None:
        """Permanently remove an entry from storage and all retriever indexes."""
        self._enforce_write(namespace)
        entry = self._storage.get(namespace, key)
        self._storage.delete(namespace, key)

        if entry is not None:
            for retriever in self._retrievers:
                retriever.delete(entry.id)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fallback_search(self, query: MemorySearch) -> list[MemoryEntry]:
        """Simple list-based search used when no retrievers are registered."""
        entries = self._storage.list(
            namespace=query.namespace,
            memory_type=query.memory_type,
        )
        if query.min_importance is not None:
            entries = [
                e for e in entries
                if e.importance is None or e.importance >= query.min_importance
            ]
        if query.metadata:
            entries = [e for e in entries if query.metadata.matches(e)]

        # Sort by importance descending when no retriever score is available
        entries.sort(key=lambda e: e.importance or 0.0, reverse=True)
        return entries[: query.limit]

    def _invalidate_conflicts(self, entry: MemoryEntry) -> None:
        candidates = self._storage.list(
            namespace=entry.namespace,
            memory_type=entry.memory_type,
        )
        for stale in find_conflicting_entries(entry, candidates):
            self._storage.put(stale.invalidate())
            for retriever in self._retrievers:
                retriever.delete(stale.id)

    def _with_related_ids(self, entry: MemoryEntry) -> MemoryEntry:
        if self._max_related_ids <= 0 or not self._retrievers:
            return entry

        search = MemorySearch(
            namespace=entry.namespace,
            query=searchable_text(entry),
            limit=self._max_related_ids + 1,
        )
        related = self.search(search)
        new_ids = [
            r.id for r in related
            if r.id != entry.id and r.id not in entry.related_ids
        ]
        all_ids = list(entry.related_ids) + new_ids
        return replace(entry, related_ids=tuple(all_ids[: self._max_related_ids]))

    def _add_reverse_links(self, entry: MemoryEntry) -> None:
        for related_id in entry.related_ids:
            related = self._storage.get_by_id(related_id)
            if related is None or related.invalidated_at is not None:
                continue
            if related.namespace != entry.namespace:
                continue
            if entry.id in related.related_ids:
                continue
            updated_ids = (entry.id, *related.related_ids)[: self._max_related_ids]
            self._storage.put(replace(related, related_ids=updated_ids))

    def _enforce_read(self, namespace: tuple[str, ...]) -> None:
        if not self._policy.can_read(namespace):
            raise NamespaceAccessError(
                f"Read access denied for namespace: {'/'.join(namespace)}"
            )

    def _enforce_write(self, namespace: tuple[str, ...]) -> None:
        if not self._policy.can_write(namespace):
            raise NamespaceAccessError(
                f"Write access denied for namespace: {'/'.join(namespace)}"
            )

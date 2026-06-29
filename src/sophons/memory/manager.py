from __future__ import annotations

from dataclasses import dataclass, field

from sophons.memory.extraction.interface import (
    MemoryExtractionRequest,
    MemoryExtractionResult,
    MemoryExtractor,
)
from sophons.memory.long_term.entry import MemoryEntry
from sophons.memory.long_term.search import MemorySearch
from sophons.memory.long_term.store import MemoryStore
from sophons.memory.reflection import MemoryReflector, ReflectionResult
from sophons.models.messages import Message


# ---------------------------------------------------------------------------
# Store configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class MemoryStoreConfig:
    """
    Wraps a ``MemoryStore`` with a name, description, and write flag.

    Multiple stores can be registered on a ``MemoryManager`` — for example a
    local in-memory store and a remote persistent store.  Reads fan-out across
    all stores; writes go only to stores where ``writable=True``.
    """

    name: str
    description: str
    store: MemoryStore
    writable: bool = True


# ---------------------------------------------------------------------------
# MemoryManager
# ---------------------------------------------------------------------------


class MemoryManager:
    """
    Orchestrates memory extraction, storage, retrieval, and reflection.

    ``MemoryManager`` is the single entry-point for long-term memory in an
    agent system.  It follows the Mem0 convention of exposing ``add`` and
    ``search`` as the primary operations and keeping reflection as a separate
    explicit step.

    Usage::

        store = MemoryStore(storage=InMemoryStorage(), retrievers=[LexicalRetriever()])
        manager = MemoryManager(
            stores=[MemoryStoreConfig(name="main", description="main store", store=store)],
            extractor=LLMMemoryExtractor(model=my_model),
        )

        # After each turn — extract and store memories from the conversation
        result = await manager.add(messages=turn_messages, namespace=("user", "alice"))

        # At the start of each turn — retrieve relevant memories
        entries = await manager.search(query=user_input, namespace=("user", "alice"))

    Args:
        stores:     One or more stores.  At least one must be provided.
        extractor:  Optional LLM extractor.  Without one ``add()`` returns an
                    empty result.
        reflector:  Optional reflector.  When provided ``add()`` calls
                    ``reflector.observe()`` after storing new entries.
    """

    def __init__(
        self,
        stores: list[MemoryStoreConfig],
        extractor: MemoryExtractor | None = None,
        reflector: MemoryReflector | None = None,
    ) -> None:
        if not stores:
            raise ValueError("MemoryManager requires at least one store.")
        self._stores = stores
        self._extractor = extractor
        self._reflector = reflector

    # ------------------------------------------------------------------
    # add — extract + store (primary write path)
    # ------------------------------------------------------------------

    async def add(
        self,
        messages: list[Message],
        namespace: tuple[str, ...],
    ) -> MemoryExtractionResult:
        """
        Extract durable memories from ``messages`` and store them.

        Follows the Mem0 ``Memory.add()`` convention: returns the extracted
        entries explicitly rather than storing silently.

        Steps:
        1. Fetch existing memories so the extractor can avoid duplicates.
        2. Run the extractor.
        3. Store new entries in all writable stores.
        4. Invalidate any keys the extractor flagged as stale.
        5. Optionally trigger reflection via ``MemoryReflector.observe()``.
        """
        if self._extractor is None:
            return MemoryExtractionResult(
                entries=[],
                skipped_reason="No extractor configured.",
            )

        primary = self._primary_store()
        existing: list[MemoryEntry] = []
        if primary is not None:
            try:
                existing = primary.list(namespace=namespace)
            except Exception:
                pass

        result = await self._extractor.extract(
            MemoryExtractionRequest(
                namespace=namespace,
                messages=messages,
                existing_memories=existing,
                memory_store=primary,
            )
        )

        for config in self._stores:
            if not config.writable:
                continue
            for entry in result.entries:
                config.store.put(entry)
            for key in result.invalidated_keys:
                config.store.invalidate(namespace, key)

        if self._reflector is not None and result.entries:
            await self._reflector.observe(result.entries, namespace)

        return result

    # ------------------------------------------------------------------
    # search — cross-store retrieval
    # ------------------------------------------------------------------

    async def search(
        self,
        query: str,
        namespace: tuple[str, ...],
        limit: int = 5,
        store_name: str | None = None,
    ) -> list[MemoryEntry]:
        """
        Search for relevant memories across all registered stores.

        When ``store_name`` is provided, only that store is queried.
        Results are deduplicated by entry ID.
        """
        if store_name is not None:
            config = next((c for c in self._stores if c.name == store_name), None)
            if config is None:
                return []
            return config.store.search(
                MemorySearch(namespace=namespace, query=query, limit=limit)
            )

        seen: set[str] = set()
        results: list[MemoryEntry] = []

        for config in self._stores:
            for entry in config.store.search(
                MemorySearch(namespace=namespace, query=query, limit=limit)
            ):
                if entry.id not in seen:
                    seen.add(entry.id)
                    results.append(entry)

        return results[:limit]

    # ------------------------------------------------------------------
    # reflect — explicit reflection pass
    # ------------------------------------------------------------------

    async def reflect(
        self,
        namespace: tuple[str, ...],
    ) -> ReflectionResult | None:
        """
        Unconditionally run a reflection pass.

        Returns ``None`` when no reflector is configured.
        """
        if self._reflector is None:
            return None
        return await self._reflector.reflect(namespace)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    async def format_context(
        self,
        query: str,
        namespace: tuple[str, ...],
        limit: int = 5,
    ) -> str:
        """
        Return a formatted string of relevant memories for prompt injection.

        Retrieves up to ``limit`` relevant entries and formats them as a
        bulleted list suitable for inclusion in a system prompt or user
        message.  Returns an empty string when no relevant memories exist.
        """
        entries = await self.search(query=query, namespace=namespace, limit=limit)
        if not entries:
            return ""
        lines = [f"- [{e.memory_type}] {e.key}: {e.content}" for e in entries]
        return "\n".join(lines)

    @property
    def store_descriptions(self) -> dict[str, str]:
        """Mapping of store name → description for all registered stores."""
        return {c.name: c.description for c in self._stores}

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _primary_store(self) -> MemoryStore | None:
        for config in self._stores:
            if config.writable:
                return config.store
        return None

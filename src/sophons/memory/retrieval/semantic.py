from __future__ import annotations

from typing import Any, Protocol

from sophons.memory.long_term.entry import MemoryEntry
from sophons.memory.long_term.search import MemorySearch, RetrievalResult
from sophons.memory.long_term.storage import MemoryStorage
from sophons.memory.long_term.text import searchable_text
from sophons.memory.retrieval._matching import (
    blend_importance,
    clamp_score,
    entry_matches_search,
)


# ---------------------------------------------------------------------------
# Protocols — implementations live in sophons.integrations
# ---------------------------------------------------------------------------


class TextEmbedder(Protocol):
    """
    Contract for a text embedding model.

    Sophons defines this contract. Bring your own implementation —
    OpenAI, Cohere, a local sentence-transformers model, etc. — and pass it
    to ``SemanticRetriever``.  Concrete adapters are provided in
    ``sophons.integrations.embedders``.
    """

    def embed(self, text: str) -> list[float]:
        """Return a dense vector representation of ``text``."""
        ...


class VectorStore(Protocol):
    """
    Contract for a vector similarity store.

    Sophons defines this contract.  Concrete adapters (Chroma, Pinecone,
    pgvector, …) are provided in ``sophons.integrations.vector_stores``.
    """

    def upsert(
        self,
        entry_id: str,
        vector: list[float],
        document: str,
        metadata: dict[str, Any],
    ) -> None:
        """Insert or replace the vector for ``entry_id``."""
        ...

    def search(
        self,
        vector: list[float],
        limit: int,
    ) -> list[VectorSearchResult]:
        """Return the ``limit`` nearest neighbours."""
        ...

    def delete(self, entry_id: str) -> None:
        """Remove the vector for ``entry_id``."""
        ...


class VectorSearchResult(Protocol):
    """A single result from a ``VectorStore.search()`` call."""

    @property
    def entry_id(self) -> str:
        """The ID of the matching ``MemoryEntry``."""
        ...

    @property
    def score(self) -> float:
        """Cosine similarity or equivalent distance score."""
        ...


# ---------------------------------------------------------------------------
# SemanticRetriever
# ---------------------------------------------------------------------------


class SemanticRetriever:
    """
    Vector-similarity retriever.

    Embeds each ``MemoryEntry`` with ``TextEmbedder`` and stores the vector in
    a ``VectorStore``.  At search time the query is embedded and the nearest
    vectors are fetched, then filtered through ``entry_matches_search()`` for
    namespace, type, and metadata constraints.

    Related entries (``entry.related_ids``) are surfaced at a slightly lower
    score to preserve graph context without a second embedding pass.

    Usage::

        retriever = SemanticRetriever(
            embedder=OpenAIEmbedder(model="text-embedding-3-small"),
            vector_store=ChromaVectorStore(collection="memory"),
            storage=my_memory_storage,
        )
        store = MemoryStore(storage=my_memory_storage, retrievers=[retriever])
    """

    def __init__(
        self,
        embedder: TextEmbedder,
        vector_store: VectorStore,
        storage: MemoryStorage,
    ) -> None:
        self._embedder = embedder
        self._vector_store = vector_store
        self._storage = storage

    # ------------------------------------------------------------------
    # MemoryRetriever interface
    # ------------------------------------------------------------------

    def add(self, entry: MemoryEntry) -> None:
        """Embed and index ``entry``."""
        document = searchable_text(entry)
        self._vector_store.upsert(
            entry_id=entry.id,
            vector=self._embedder.embed(document),
            document=document,
            metadata={
                "memory_type": entry.memory_type,
                "namespace": "/".join(entry.namespace),
                **entry.metadata,
            },
        )

    def search(self, search: MemorySearch) -> list[RetrievalResult]:
        """Return entries ranked by vector similarity to ``search.query``."""
        query_vector = self._embedder.embed(search.query)
        # Over-fetch so we have room to filter by namespace/type/metadata
        vector_results = self._vector_store.search(
            vector=query_vector,
            limit=max(search.limit * 10, search.limit),
        )

        results_by_id: dict[str, RetrievalResult] = {}

        for vr in vector_results:
            if vr.score <= 0:
                continue

            entry = self._storage.get_by_id(vr.entry_id)
            if entry is None or not entry_matches_search(entry, search):
                continue

            relevance = clamp_score(vr.score)
            results_by_id[entry.id] = RetrievalResult(
                entry_id=entry.id,
                source="semantic",
                score=blend_importance(relevance, entry),
                relevance_score=relevance,
                importance_score=entry.importance,
                reason="vector cosine similarity",
            )

            # Surface related entries at a discounted score
            self._add_related(entry, relevance, search, results_by_id)

            if len(results_by_id) >= search.limit:
                break

        sorted_results = sorted(
            results_by_id.values(), key=lambda r: r.score, reverse=True
        )
        return sorted_results[: search.limit]

    def delete(self, entry_id: str) -> None:
        """Remove an entry from the vector index."""
        self._vector_store.delete(entry_id)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _add_related(
        self,
        parent: MemoryEntry,
        parent_relevance: float,
        search: MemorySearch,
        results_by_id: dict[str, RetrievalResult],
    ) -> None:
        discounted = parent_relevance * 0.9
        for related_id in parent.related_ids:
            if related_id in results_by_id:
                continue
            related = self._storage.get_by_id(related_id)
            if related is None or not entry_matches_search(related, search):
                continue
            results_by_id[related.id] = RetrievalResult(
                entry_id=related.id,
                source="semantic",
                score=blend_importance(discounted, related),
                relevance_score=discounted,
                importance_score=related.importance,
                reason=f"one-hop related entry from semantic match {parent.id}",
            )

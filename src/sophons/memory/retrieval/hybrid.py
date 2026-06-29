from __future__ import annotations

from sophons.memory.long_term.entry import MemoryEntry, MemoryType
from sophons.memory.long_term.ranking import reciprocal_rank_score
from sophons.memory.long_term.search import MemorySearch, RetrievalResult
from sophons.memory.long_term.store import MemoryRetriever


class HybridRetriever:
    """
    Combines multiple ``MemoryRetriever`` instances via Reciprocal Rank Fusion.

    Results from every registered retriever are merged: each retriever
    contributes a ranked list and their RRF scores are summed per entry ID.
    The entry with the highest combined score is returned first.

    Optionally a ``routes`` mapping allows routing specific ``MemoryType``
    values to a dedicated subset of retrievers::

        hybrid = HybridRetriever(
            retrievers=[lexical, semantic],
            routes={
                "episodic": [episodic_retriever],
                None: [lexical, semantic],   # default for all other types
            },
        )

    When ``routes`` is provided and a type has no explicit entry the ``None``
    key is used as the fallback; if that is also absent, all retrievers are
    used.
    """

    def __init__(
        self,
        retrievers: list[MemoryRetriever],
        routes: dict[MemoryType | None, list[MemoryRetriever]] | None = None,
    ) -> None:
        self._retrievers = retrievers
        self._routes = routes

    # ------------------------------------------------------------------
    # MemoryRetriever interface
    # ------------------------------------------------------------------

    def add(self, entry: MemoryEntry) -> None:
        for retriever in self._retrievers:
            retriever.add(entry)

    def search(self, search: MemorySearch) -> list[RetrievalResult]:
        rrf_scores: dict[str, float] = {}
        best_result: dict[str, RetrievalResult] = {}

        for retriever in self._select_retrievers(search):
            results = retriever.search(search)
            for rank, result in enumerate(results, start=1):
                rrf_scores[result.entry_id] = (
                    rrf_scores.get(result.entry_id, 0.0) + reciprocal_rank_score(rank)
                )
                existing = best_result.get(result.entry_id)
                if existing is None or result.score > existing.score:
                    best_result[result.entry_id] = result

        fused = [
            RetrievalResult(
                entry_id=entry_id,
                source=best_result[entry_id].source,
                score=rrf_score,
                relevance_score=best_result[entry_id].relevance_score,
                recency_score=best_result[entry_id].recency_score,
                importance_score=best_result[entry_id].importance_score,
                reason=best_result[entry_id].reason,
            )
            for entry_id, rrf_score in rrf_scores.items()
        ]
        fused.sort(key=lambda r: r.score, reverse=True)
        return fused[: search.limit]

    def delete(self, entry_id: str) -> None:
        for retriever in self._retrievers:
            retriever.delete(entry_id)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _select_retrievers(self, search: MemorySearch) -> list[MemoryRetriever]:
        if self._routes is None:
            return self._retrievers
        if search.memory_type in self._routes:
            return self._routes[search.memory_type]
        return self._routes.get(None, self._retrievers)

from __future__ import annotations

from sophons.documents.base import Document
from sophons.memory.long_term.entry import MemoryEntry
from sophons.memory.long_term.search import MemorySearch, RetrievalResult
from sophons.memory.long_term.text import searchable_text
from sophons.memory.retrieval._matching import (
    blend_importance,
    entry_matches_search,
    token_overlap_score,
)
from sophons.retrieval.bm25 import BM25Retriever


class LexicalRetriever:
    """
    BM25-based lexical retriever.

    Indexes every ``MemoryEntry`` as a bag-of-words document built from
    ``searchable_text()``.  On search it scores all candidate entries with
    BM25 Okapi, falls back to raw token-overlap when BM25 yields no hits,
    and blends the resulting relevance score with the entry's importance.

    This is the only retriever shipped in the core library — it has no
    external dependencies beyond ``sophons.retrieval.bm25``.  For embedding-
    based retrieval see ``SemanticRetriever`` in this package.
    """

    def __init__(self) -> None:
        self._entries: dict[str, MemoryEntry] = {}

    # ------------------------------------------------------------------
    # MemoryRetriever interface
    # ------------------------------------------------------------------

    def add(self, entry: MemoryEntry) -> None:
        """Index ``entry`` so it is searchable."""
        self._entries[entry.id] = entry

    def search(self, search: MemorySearch) -> list[RetrievalResult]:
        """Return entries ranked by BM25 relevance to ``search.query``."""
        candidates: list[MemoryEntry] = [
            e for e in self._entries.values() if entry_matches_search(e, search)
        ]

        if not candidates:
            return []

        # Build a temporary BM25 index from the candidate documents
        docs = [
            Document(id=entry.id, content=searchable_text(entry))
            for entry in candidates
        ]
        retriever = BM25Retriever(docs)
        ranked_docs = retriever.retrieve(search.query, limit=search.limit)

        if ranked_docs:
            # BM25 returned hits — normalise scores and build results
            max_score = max(d.score or 0.0 for d in ranked_docs)
            normalise = (lambda s: s / max_score) if max_score > 0 else (lambda s: s)

            results: list[RetrievalResult] = []
            for doc in ranked_docs:
                entry = self._entries.get(doc.id or "")
                if entry is None:
                    continue
                raw_score = normalise(doc.score or 0.0)
                results.append(
                    RetrievalResult(
                        entry_id=entry.id,
                        source="lexical",
                        score=blend_importance(raw_score, entry),
                        relevance_score=raw_score,
                        importance_score=entry.importance,
                        reason="BM25 lexical search over searchable_text()",
                    )
                )
            return results

        # Fall back to token-overlap when BM25 finds nothing
        scored: list[tuple[float, MemoryEntry]] = []
        for entry in candidates:
            s = token_overlap_score(search.query, [searchable_text(entry)])
            if s > 0:
                scored.append((s, entry))

        scored.sort(key=lambda se: se[0], reverse=True)
        return [
            RetrievalResult(
                entry_id=entry.id,
                source="lexical",
                score=blend_importance(score, entry),
                relevance_score=score,
                importance_score=entry.importance,
                reason="token-overlap fallback (no BM25 hits)",
            )
            for score, entry in scored[: search.limit]
        ]

    def delete(self, entry_id: str) -> None:
        """Remove an entry from the index."""
        self._entries.pop(entry_id, None)

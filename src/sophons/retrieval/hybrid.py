from __future__ import annotations

from opentelemetry import trace

from sophons.documents import Document
from sophons.observability import _semconv
from sophons.retrieval.base import Retriever

_TRACER = trace.get_tracer("sophons.retrieval")

_RRF_K = 60  # standard damping constant from the RRF paper


class HybridRetriever:
    """
    Fuses the results of several retrievers with Reciprocal Rank Fusion.

    Dense retrieval misses exact identifiers; sparse retrieval misses
    paraphrase. Each is strong where the other is blind — so ask both,
    then fuse by *rank*, not score: BM25 scores and cosine similarities
    live on incomparable scales, but "third place on this list" always
    means the same thing.

        rrf_score(doc) = Σ  1 / (k + rank_in_each_list)

    A document ranked well by both retrievers beats one ranked first by
    a single retriever and absent from the other — fusion rewards
    consensus.

    Args:
        retrievers: Any number of ``Retriever`` protocol members —
                    typically ``[BM25Retriever(...), SemanticRetriever(...)]``.
        k:          RRF damping constant. Bigger k flattens rank
                    differences; 60 is the standard default.
    """

    def __init__(self, retrievers: list[Retriever], *, k: int = _RRF_K) -> None:
        if not retrievers:
            raise ValueError("HybridRetriever needs at least one retriever.")
        self._retrievers = retrievers
        self._k = k

    def retrieve(self, query: str, *, limit: int = 10) -> list[Document]:
        with _TRACER.start_as_current_span(
            "retriever.search",
            attributes={_semconv.RETRIEVER: "hybrid", _semconv.LIMIT: limit},
        ) as span:
            # RRF needs depth to fuse well: ask each retriever for more
            # than the caller wants, then fuse down to `limit`.
            fetch = max(limit * 4, 10)

            scores: dict[str, float] = {}
            documents: dict[str, Document] = {}
            for retriever in self._retrievers:
                results = retriever.retrieve(query, limit=fetch)
                for rank, document in enumerate(results):
                    scores[document.id] = scores.get(document.id, 0.0) + 1.0 / (
                        self._k + rank + 1
                    )
                    documents.setdefault(document.id, document)

            fused = sorted(scores, key=lambda doc_id: scores[doc_id], reverse=True)
            top = [documents[doc_id] for doc_id in fused[:limit]]
            span.set_attribute(_semconv.RESULT_COUNT, len(top))
            return top

from __future__ import annotations

from opentelemetry import trace

from sophons.documents import Document
from sophons.models.embeddings import EmbeddingModel
from sophons.observability import _semconv
from sophons.retrieval.base import VectorStore

_TRACER = trace.get_tracer("sophons.retrieval")


class SemanticRetriever:
    """
    Vector-similarity retriever for Documents.

    Combines an ``EmbeddingModel`` with a ``VectorStore`` to implement
    semantic search over a document corpus. At index time each document is
    embedded and stored. At search time the query is embedded and the nearest
    neighbours are returned.

    Args:
        embedder:     Any ``EmbeddingModel`` — ``OpenAIEmbeddings``,
                      ``SentenceTransformerEmbeddings``, or your own.
        vector_store: Any ``VectorStore`` — ``InMemoryVectorStore``,
                      ``ChromaVectorStore``, or your own.

    Usage::

        retriever = SemanticRetriever(
            embedder=OpenAIEmbeddings(api_key="sk-..."),
            vector_store=InMemoryVectorStore(),
        )
        retriever.add(documents)
        results = retriever.retrieve("what is the refund policy?", limit=3)
    """

    def __init__(
        self,
        embedder: EmbeddingModel,
        vector_store: VectorStore,
    ) -> None:
        self._embedder = embedder
        self._store = vector_store

    def add(self, documents: list[Document]) -> None:
        """Embed and index a list of documents."""
        if not documents:
            return
        texts = [doc.content for doc in documents]
        vectors = self._embedder.embed_documents(texts)
        self._store.add(documents, vectors)

    def retrieve(self, query: str, *, limit: int = 10) -> list[Document]:
        """Return the most semantically similar documents to ``query``."""
        with _TRACER.start_as_current_span(
            "retriever.search",
            attributes={_semconv.RETRIEVER: "semantic", _semconv.LIMIT: limit},
        ) as span:
            vector = self._embedder.embed_query(query)
            results = self._store.search(vector, limit=limit)
            span.set_attribute(_semconv.RESULT_COUNT, len(results))
            return results

from __future__ import annotations

import math

from sophons.documents import Document


class InMemoryVectorStore:
    """
    Simple in-memory vector store using cosine similarity.

    No external dependencies. Suitable for small corpora (up to ~10k documents)
    and for development and testing. For larger corpora use ``ChromaVectorStore``.

    Usage::

        store = InMemoryVectorStore()
        store.add(documents, vectors)
        results = store.search(query_vector, limit=5)
    """

    def __init__(self) -> None:
        self._documents: list[Document] = []
        self._vectors: list[list[float]] = []

    def add(self, documents: list[Document], vectors: list[list[float]]) -> None:
        """Store documents alongside their embedding vectors."""
        self._documents.extend(documents)
        self._vectors.extend(vectors)

    def search(self, vector: list[float], *, limit: int = 10) -> list[Document]:
        """Return the ``limit`` most similar documents by cosine similarity."""
        if not self._documents:
            return []

        scored = [
            (self._cosine(vector, v), doc)
            for v, doc in zip(self._vectors, self._documents)
        ]
        scored.sort(key=lambda item: item[0], reverse=True)
        return [doc.with_score(score) for score, doc in scored[:limit]]

    def delete(self, ids: list[str]) -> None:
        """Remove documents by their IDs."""
        id_set = set(ids)
        pairs = [
            (v, doc)
            for v, doc in zip(self._vectors, self._documents)
            if doc.id not in id_set
        ]
        if pairs:
            self._vectors, self._documents = zip(*pairs)  # type: ignore[assignment]
            self._vectors = list(self._vectors)
            self._documents = list(self._documents)
        else:
            self._vectors = []
            self._documents = []

    def __len__(self) -> int:
        return len(self._documents)

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (norm_a * norm_b)

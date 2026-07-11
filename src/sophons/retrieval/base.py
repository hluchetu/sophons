from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from sophons.documents import Document


@runtime_checkable
class Retriever(Protocol):
    """Sync retriever contract: query text in, documents out."""

    def retrieve(self, query: str, *, limit: int = 10) -> list[Document]:
        ...


@runtime_checkable
class AsyncRetriever(Protocol):
    """Async retriever contract: query text in, documents out."""

    async def retrieve(self, query: str, *, limit: int = 10) -> list[Document]:
        ...


@runtime_checkable
class VectorStore(Protocol):
    """
    Contract for a vector similarity store scoped to Documents.

    Concrete implementations live in ``sophons.integrations.vector_stores``.
    """

    def add(self, documents: list[Document], vectors: list[list[float]]) -> None:
        """Store documents alongside their pre-computed embedding vectors."""
        ...

    def search(self, vector: list[float], *, limit: int = 10) -> list[Document]:
        """Return the ``limit`` most similar documents to ``vector``."""
        ...

    def delete(self, ids: list[str]) -> None:
        """Remove documents by their IDs."""
        ...

from __future__ import annotations

from typing import Protocol, runtime_checkable

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

from __future__ import annotations

from typing import Protocol, runtime_checkable


Vector = list[float]


@runtime_checkable
class EmbeddingModel(Protocol):
    """Sync embedding model contract: text in, vectors out."""

    def embed_query(self, text: str) -> Vector:
        ...

    def embed_documents(self, texts: list[str]) -> list[Vector]:
        ...


@runtime_checkable
class AsyncEmbeddingModel(Protocol):
    """Async embedding model contract: text in, vectors out."""

    async def embed_query(self, text: str) -> Vector:
        ...

    async def embed_documents(self, texts: list[str]) -> list[Vector]:
        ...

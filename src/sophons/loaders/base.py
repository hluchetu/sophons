from __future__ import annotations

from collections.abc import AsyncIterable, Iterable
from typing import Protocol, runtime_checkable

from sophons.documents import Document


@runtime_checkable
class Loader(Protocol):
    """Sync loader contract: external source in, documents out."""

    def load(self) -> list[Document]:
        ...

    def lazy_load(self) -> Iterable[Document]:
        ...


@runtime_checkable
class AsyncLoader(Protocol):
    """Async loader contract: external source in, documents out."""

    async def load(self) -> list[Document]:
        ...

    def lazy_load(self) -> AsyncIterable[Document]:
        ...

from __future__ import annotations

import pytest

from sophons.documents import Document
from sophons.loaders import AsyncLoader, Loader


class StaticLoader:
    def __init__(self, documents: list[Document]) -> None:
        self._documents = documents

    def load(self) -> list[Document]:
        return list(self._documents)

    def lazy_load(self):
        yield from self._documents


class StaticAsyncLoader:
    def __init__(self, documents: list[Document]) -> None:
        self._documents = documents

    async def load(self) -> list[Document]:
        return list(self._documents)

    async def lazy_load(self):
        for document in self._documents:
            yield document


def test_loader_protocol_accepts_matching_object() -> None:
    loader = StaticLoader([Document(content="hello")])

    assert isinstance(loader, Loader)
    assert loader.load() == [Document(content="hello")]
    assert list(loader.lazy_load()) == [Document(content="hello")]


@pytest.mark.asyncio
async def test_async_loader_protocol_accepts_matching_object() -> None:
    loader = StaticAsyncLoader([Document(content="hello")])

    assert isinstance(loader, AsyncLoader)
    assert await loader.load() == [Document(content="hello")]
    assert [document async for document in loader.lazy_load()] == [
        Document(content="hello")
    ]

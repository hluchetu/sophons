from __future__ import annotations

import pytest

from sophons.documents import Document
from sophons.retrieval import AsyncRetriever, Retriever


class StaticRetriever:
    def __init__(self, documents: list[Document]) -> None:
        self._documents = documents

    def retrieve(self, query: str, *, limit: int = 10) -> list[Document]:
        return self._documents[:limit]


class StaticAsyncRetriever:
    def __init__(self, documents: list[Document]) -> None:
        self._documents = documents

    async def retrieve(self, query: str, *, limit: int = 10) -> list[Document]:
        return self._documents[:limit]


def test_retriever_protocol_accepts_matching_object() -> None:
    retriever = StaticRetriever([Document(content="hello")])

    assert isinstance(retriever, Retriever)
    assert retriever.retrieve("hello") == [Document(content="hello")]


@pytest.mark.asyncio
async def test_async_retriever_protocol_accepts_matching_object() -> None:
    retriever = StaticAsyncRetriever([Document(content="hello")])

    assert isinstance(retriever, AsyncRetriever)
    assert await retriever.retrieve("hello") == [Document(content="hello")]

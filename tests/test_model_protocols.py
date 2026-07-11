from __future__ import annotations

import pytest

from sophons.models import (
    AsyncChatModel,
    AsyncEmbeddingModel,
    ChatModel,
    EmbeddingModel,
    Message,
)


class EchoChatModel:
    def invoke(self, messages: list[Message]) -> Message:
        return Message(role="assistant", content=messages[-1].content)


class AsyncEchoChatModel:
    async def invoke(self, messages: list[Message]) -> Message:
        return Message(role="assistant", content=messages[-1].content)


class LengthEmbeddingModel:
    def embed_query(self, text: str) -> list[float]:
        return [float(len(text))]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]


class AsyncLengthEmbeddingModel:
    async def embed_query(self, text: str) -> list[float]:
        return [float(len(text))]

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [await self.embed_query(text) for text in texts]


def test_chat_model_protocol_accepts_matching_object() -> None:
    model = EchoChatModel()

    assert isinstance(model, ChatModel)
    assert model.invoke([Message(role="user", content="hello")]) == Message(
        role="assistant",
        content="hello",
    )


@pytest.mark.asyncio
async def test_async_chat_model_protocol_accepts_matching_object() -> None:
    model = AsyncEchoChatModel()

    assert isinstance(model, AsyncChatModel)
    assert await model.invoke([Message(role="user", content="hello")]) == Message(
        role="assistant",
        content="hello",
    )


def test_embedding_model_protocol_accepts_matching_object() -> None:
    model = LengthEmbeddingModel()

    assert isinstance(model, EmbeddingModel)
    assert model.embed_query("hello") == [5.0]
    assert model.embed_documents(["hi", "hello"]) == [[2.0], [5.0]]


@pytest.mark.asyncio
async def test_async_embedding_model_protocol_accepts_matching_object() -> None:
    model = AsyncLengthEmbeddingModel()

    assert isinstance(model, AsyncEmbeddingModel)
    assert await model.embed_query("hello") == [5.0]
    assert await model.embed_documents(["hi", "hello"]) == [[2.0], [5.0]]

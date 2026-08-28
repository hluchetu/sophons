from __future__ import annotations

from typing import Any

from sophons.models.embeddings import Vector


class OpenAIEmbeddings:
    """Synchronous OpenAI embedding model."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "text-embedding-3-small",
        dimensions: int | None = None,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "OpenAIEmbeddings requires the 'openai' extra. "
                "Install it with: uv sync --extra openai"
            ) from exc

        self._model = model
        self._dimensions = dimensions
        self._client = OpenAI(api_key=api_key)

    def embed_query(self, text: str) -> Vector:
        return self.embed_documents([text])[0]

    def embed_documents(self, texts: list[str]) -> list[Vector]:
        if not texts:
            return []

        kwargs: dict[str, Any] = {
            "input": texts,
            "model": self._model,
        }

        if self._dimensions is not None:
            kwargs["dimensions"] = self._dimensions

        response = self._client.embeddings.create(**kwargs)
        return [item.embedding for item in response.data]


class AsyncOpenAIEmbeddings:
    """Asynchronous OpenAI embedding model."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "text-embedding-3-small",
        dimensions: int | None = None,
    ) -> None:
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise ImportError(
                "AsyncOpenAIEmbeddings requires the 'openai' extra. "
                "Install it with: uv sync --extra openai"
            ) from exc

        self._model = model
        self._dimensions = dimensions
        self._client = AsyncOpenAI(api_key=api_key)

    async def embed_query(self, text: str) -> Vector:
        vectors = await self.embed_documents([text])
        return vectors[0]

    async def embed_documents(self, texts: list[str]) -> list[Vector]:
        if not texts:
            return []

        kwargs: dict[str, Any] = {
            "input": texts,
            "model": self._model,
        }

        if self._dimensions is not None:
            kwargs["dimensions"] = self._dimensions

        response = await self._client.embeddings.create(**kwargs)
        return [item.embedding for item in response.data]

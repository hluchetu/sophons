from __future__ import annotations

from openai import AsyncOpenAI, OpenAI

from sophons.models.embeddings import Vector


class OpenAIEmbeddings:
    """
    OpenAI embedding model.

    Implements both ``EmbeddingModel`` (sync) and ``AsyncEmbeddingModel``
    (async) protocols from ``sophons.models.embeddings``.

    Args:
        api_key:    OpenAI API key.
        model:      Embedding model name. Defaults to ``text-embedding-3-small``.
        dimensions: Optional output dimension (supported by v3 models).

    Usage::

        embedder = OpenAIEmbeddings(api_key="sk-...")
        vector = embedder.embed_query("what is the refund policy?")
        vectors = embedder.embed_documents(["doc one", "doc two"])
    """

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        dimensions: int | None = None,
    ) -> None:
        self._model = model
        self._dimensions = dimensions
        self._client = OpenAI(api_key=api_key)
        self._async_client = AsyncOpenAI(api_key=api_key)

    # ------------------------------------------------------------------
    # Sync (EmbeddingModel protocol)
    # ------------------------------------------------------------------

    def embed_query(self, text: str) -> Vector:
        """Embed a single query string."""
        return self._embed([text])[0]

    def embed_documents(self, texts: list[str]) -> list[Vector]:
        """Embed a batch of document strings."""
        return self._embed(texts)

    # ------------------------------------------------------------------
    # Async (AsyncEmbeddingModel protocol)
    # ------------------------------------------------------------------

    async def async_embed_query(self, text: str) -> Vector:
        """Async embed a single query string."""
        results = await self._async_embed([text])
        return results[0]

    async def async_embed_documents(self, texts: list[str]) -> list[Vector]:
        """Async embed a batch of document strings."""
        return await self._async_embed(texts)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _embed(self, texts: list[str]) -> list[Vector]:
        kwargs: dict = dict(input=texts, model=self._model)
        if self._dimensions:
            kwargs["dimensions"] = self._dimensions
        response = self._client.embeddings.create(**kwargs)
        return [item.embedding for item in response.data]

    async def _async_embed(self, texts: list[str]) -> list[Vector]:
        kwargs: dict = dict(input=texts, model=self._model)
        if self._dimensions:
            kwargs["dimensions"] = self._dimensions
        response = await self._async_client.embeddings.create(**kwargs)
        return [item.embedding for item in response.data]

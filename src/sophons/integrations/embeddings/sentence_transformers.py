from __future__ import annotations

from typing import Any

from sophons.models.embeddings import Vector


class SentenceTransformerEmbeddings:
    """Local Sentence Transformers embedding model."""

    def __init__(
        self,
        *,
        model: str = "all-MiniLM-L6-v2",
        device: str = "cpu",
        normalize: bool = False,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "SentenceTransformerEmbeddings requires the "
                "'sentence-transformers' extra. Install it with: "
                "uv sync --extra sentence-transformers"
            ) from exc

        self._model: Any = SentenceTransformer(model, device=device)
        self._normalize = normalize

    def embed_query(self, text: str) -> Vector:
        return self._model.encode(
            text,
            normalize_embeddings=self._normalize,
        ).tolist()

    def embed_documents(self, texts: list[str]) -> list[Vector]:
        if not texts:
            return []

        return self._model.encode(
            texts,
            normalize_embeddings=self._normalize,
        ).tolist()

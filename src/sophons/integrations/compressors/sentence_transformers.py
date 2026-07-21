from __future__ import annotations

from importlib import import_module
from typing import Any, cast

from sophons.documents import Document
from sophons.errors import ConfigurationError, MissingDependencyError


class CrossEncoderReranker:
    """Rerank retrieved documents with a sentence-transformers CrossEncoder."""

    def __init__(
        self,
        *,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_n: int = 3,
        batch_size: int = 16,
        max_chars: int = 1_200,
        device: str | None = None,
    ) -> None:
        if top_n < 0:
            raise ConfigurationError(
                "top_n must be greater than or equal to 0.",
                details={"parameter": "top_n", "value": top_n},
            )
        if batch_size <= 0:
            raise ConfigurationError(
                "batch_size must be greater than 0.",
                details={"parameter": "batch_size", "value": batch_size},
            )
        if max_chars <= 0:
            raise ConfigurationError(
                "max_chars must be greater than 0.",
                details={"parameter": "max_chars", "value": max_chars},
            )

        try:
            module = import_module("sentence_transformers")
        except ImportError as exc:
            raise MissingDependencyError(
                "sentence-transformers is required for CrossEncoderReranker. "
                "Install it with: pip install 'sophons[sentence-transformers]'"
            ) from exc

        cross_encoder = getattr(module, "CrossEncoder")
        self.model_name = model_name
        self.model: Any = cross_encoder(model_name, device=device)
        self.top_n = top_n
        self.batch_size = batch_size
        self.max_chars = max_chars

    async def compress(self, documents: list[Document], query: str) -> list[Document]:
        pairs = [(query, document.content[: self.max_chars]) for document in documents]
        scores = cast(
            list[float],
            self.model.predict(
                pairs,
                batch_size=self.batch_size,
                convert_to_numpy=False,
                show_progress_bar=False,
            ),
        )

        ranked = sorted(
            zip(documents, scores, strict=True),
            key=lambda item: float(item[1]),
            reverse=True,
        )

        return [
            document.with_score(float(score)).with_metadata(reranker=self.model_name)
            for document, score in ranked[: self.top_n]
        ]

from __future__ import annotations

from typing import Any

from sophons.documents import Document
from sophons.errors import MissingDependencyError


class NumPyVectorStore:
    """Exact in-memory cosine search over a NumPy float32 matrix.

    Documents remain in insertion order and vector row ``n`` always belongs to
    document ``n``. The store is a runtime index; persistence belongs to the
    consuming application.
    """

    def __init__(self) -> None:
        try:
            import numpy as np
        except ImportError as exc:
            raise MissingDependencyError(
                "NumPyVectorStore requires the 'numpy' extra. "
                "Install it with: uv sync --extra numpy"
            ) from exc

        self._np = np
        self._documents: list[Document] = []
        self._vectors: Any | None = None

    def add(self, documents: list[Document], vectors: list[list[float]]) -> None:
        """Add documents using the common ``VectorStore`` protocol."""
        self.add_matrix(documents, vectors)

    def add_matrix(self, documents: list[Document], vectors: Any) -> None:
        """Add documents from a two-dimensional array without list conversion."""
        matrix = self._np.asarray(vectors, dtype=self._np.float32)
        if matrix.ndim != 2:
            raise ValueError(f"vectors must be a 2-D matrix, got shape {matrix.shape}")
        if matrix.shape[0] != len(documents):
            raise ValueError(
                f"received {matrix.shape[0]} vectors for {len(documents)} documents"
            )
        if not self._np.isfinite(matrix).all():
            raise ValueError("vectors contain non-finite values")

        if self._vectors is None:
            self._vectors = matrix
        else:
            if matrix.shape[1] != self._vectors.shape[1]:
                raise ValueError(
                    f"vector dimension {matrix.shape[1]} does not match "
                    f"existing dimension {self._vectors.shape[1]}"
                )
            if matrix.shape[0]:
                self._vectors = self._np.vstack((self._vectors, matrix))

        self._documents.extend(documents)

    def search(self, vector: list[float], *, limit: int = 10) -> list[Document]:
        """Return the most similar documents using exact cosine similarity."""
        if not self._documents or limit <= 0:
            return []

        query = self._np.asarray(vector, dtype=self._np.float32)
        if query.ndim != 1:
            raise ValueError(f"query must be a 1-D vector, got shape {query.shape}")
        if query.shape[0] != self._vectors.shape[1]:
            raise ValueError(
                f"query dimension {query.shape[0]} does not match "
                f"store dimension {self._vectors.shape[1]}"
            )
        if not self._np.isfinite(query).all():
            raise ValueError("query contains non-finite values")

        numerators = self._vectors @ query
        denominators = self._np.linalg.norm(self._vectors, axis=1) * self._np.linalg.norm(
            query
        )
        scores = self._np.divide(
            numerators,
            denominators,
            out=self._np.zeros(len(self._documents), dtype=self._np.float32),
            where=denominators != 0,
        )
        count = min(limit, len(self._documents))
        indices = self._np.argsort(-scores, kind="stable")[:count]
        return [
            self._documents[int(index)].with_score(float(scores[index]))
            for index in indices
        ]

    def delete(self, ids: list[str]) -> None:
        """Remove documents and their corresponding matrix rows by ID."""
        if not ids or not self._documents:
            return

        removed = set(ids)
        keep = self._np.asarray(
            [document.id not in removed for document in self._documents],
            dtype=bool,
        )
        self._documents = [
            document
            for document, should_keep in zip(self._documents, keep, strict=True)
            if should_keep
        ]
        self._vectors = self._vectors[keep]

    def __len__(self) -> int:
        return len(self._documents)

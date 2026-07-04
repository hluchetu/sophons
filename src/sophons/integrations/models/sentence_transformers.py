from __future__ import annotations

from sophons.models.embeddings import Vector


class SentenceTransformerEmbeddings:
    """
    Local embedding model using sentence-transformers.

    Runs entirely on device — no API key, no internet connection required.
    Implements the ``EmbeddingModel`` protocol from ``sophons.models.embeddings``.

    Requires ``sentence-transformers`` to be installed::

        pip install sentence-transformers

    Args:
        model:      Model name from HuggingFace Hub. Defaults to
                    ``all-MiniLM-L6-v2`` (90 MB, fast, good quality).
                    Other good options: ``bge-small-en-v1.5``, ``all-mpnet-base-v2``.
        device:     Device to run on: ``"cpu"``, ``"cuda"``, ``"mps"``.
                    Defaults to ``"cpu"``.

    Usage::

        embedder = SentenceTransformerEmbeddings()
        vector = embedder.embed_query("what is the refund policy?")
        vectors = embedder.embed_documents(["doc one", "doc two"])
    """

    def __init__(
        self,
        model: str = "all-MiniLM-L6-v2",
        device: str = "cpu",
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for SentenceTransformerEmbeddings. "
                "Install it with: pip install sentence-transformers"
            ) from exc

        self._model = SentenceTransformer(model, device=device)

    # ------------------------------------------------------------------
    # EmbeddingModel protocol
    # ------------------------------------------------------------------

    def embed_query(self, text: str) -> Vector:
        """Embed a single query string."""
        return self._model.encode(text).tolist()

    def embed_documents(self, texts: list[str]) -> list[Vector]:
        """Embed a batch of document strings."""
        return self._model.encode(texts).tolist()

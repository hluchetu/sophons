from __future__ import annotations

from typing import Protocol

from sophons.documents.base import Document


class DocumentCompressor(Protocol):
    """
    Contract for re-ranking or filtering retrieved documents.

    Follows LangChain's ``BaseDocumentCompressor`` convention — compressors
    sit between a retriever and the LLM, pruning or reordering the retrieved
    documents to improve answer quality.

    Common implementations:

    - **Cross-encoder reranker** — scores every (query, document) pair with a
      dedicated model (e.g. ``cross-encoder/ms-marco-MiniLM-L-6-v2``).
    - **LLM-based filter** — asks the LLM whether each document is relevant.
    - **MMR (Maximum Marginal Relevance)** — balances relevance with diversity.

    Concrete implementations go in ``sophons.integrations.compressors``.

    To use a custom compressor implement this Protocol and pass it to
    ``RAGPipeline``::

        class MyReranker:
            async def compress(
                self, documents: list[Document], query: str
            ) -> list[Document]: ...

        pipeline = RAGPipeline(..., compressor=MyReranker())
    """

    async def compress(
        self,
        documents: list[Document],
        query: str,
    ) -> list[Document]:
        """
        Return a reordered / filtered subset of ``documents``.

        The returned list may be shorter than the input (filtered) and must
        be ordered by descending relevance to ``query``.
        """
        ...

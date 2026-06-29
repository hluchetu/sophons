from __future__ import annotations

from dataclasses import dataclass, field

from sophons.documents.base import Document
from sophons.rag.splitters.fixed import FixedSizeTextSplitter
from sophons.rag.splitters.interface import TextSplitter


@dataclass(frozen=True)
class ParentDocumentChunks:
    """
    Output of ``ParentDocumentSplitter``.

    ``parents`` maps a parent chunk ID to its text.
    ``children`` are the small child chunks that get embedded — each carries
    a ``parent_id`` in its metadata so the full parent can be fetched at
    retrieval time.
    """

    parents: dict[str, Document]
    children: list[Document]


class ParentDocumentSplitter:
    """
    Two-level splitting: small child chunks for embedding, large parent chunks
    for retrieval context.

    The child chunks are indexed and searched by the retriever.  When a child
    matches a query the full parent chunk is returned instead, giving the LLM
    much richer context while keeping embeddings precise.

    This is the same pattern as LangChain's ``ParentDocumentRetriever``.

    Args:
        parent_splitter: Splitter for the large parent chunks.
                         Defaults to ``FixedSizeTextSplitter(500)``.
        child_splitter:  Splitter for the small child chunks.
                         Defaults to ``FixedSizeTextSplitter(200)``.

    Usage::

        splitter = ParentDocumentSplitter()
        result = splitter.split_with_parents(document)
        # embed result.children, store result.parents
    """

    def __init__(
        self,
        parent_splitter: TextSplitter | None = None,
        child_splitter: TextSplitter | None = None,
    ) -> None:
        self._parent_splitter = parent_splitter or FixedSizeTextSplitter(
            chunk_size=500, overlap=0
        )
        self._child_splitter = child_splitter or FixedSizeTextSplitter(
            chunk_size=200, overlap=20
        )

    def split(self, document: Document) -> list[Document]:
        """Return only the child chunks (satisfies ``TextSplitter`` Protocol)."""
        return self.split_with_parents(document).children

    def split_with_parents(self, document: Document) -> ParentDocumentChunks:
        """Return both parent and child chunks."""
        parent_chunks = self._parent_splitter.split(document)
        parents: dict[str, Document] = {}
        children: list[Document] = []

        for parent in parent_chunks:
            parent_id = parent.id or f"parent_{len(parents)}"
            parents[parent_id] = Document(
                content=parent.content,
                metadata={**parent.metadata, "parent_id": parent_id},
                id=parent_id,
            )
            for child in self._child_splitter.split(parent):
                children.append(
                    Document(
                        content=child.content,
                        metadata={**child.metadata, "parent_id": parent_id},
                        id=child.id,
                    )
                )

        return ParentDocumentChunks(parents=parents, children=children)

from __future__ import annotations

from typing import Protocol

from sophons.documents.base import Document


class TextSplitter(Protocol):
    """
    Contract for text splitting strategies.

    A ``TextSplitter`` takes a ``Document`` and returns a list of smaller
    ``Document`` objects, each inheriting the parent's metadata plus a
    ``chunk_index`` field.

    Sophons follows LangChain's convention of separating splitting from
    loading — loaders produce ``Document`` objects, splitters chunk them.

    To use a custom splitter implement this Protocol in your own class and
    pass it to ``RAGPipeline``::

        class MySplitter:
            def split(self, document: Document) -> list[Document]: ...

        pipeline = RAGPipeline(splitter=MySplitter(), ...)
    """

    def split(self, document: Document) -> list[Document]:
        """Split ``document`` into smaller ``Document`` chunks."""
        ...

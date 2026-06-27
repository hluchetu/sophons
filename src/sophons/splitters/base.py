from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol, runtime_checkable

from sophons.documents import Document


@runtime_checkable
class Splitter(Protocol):
    """Splitter contract: documents in, smaller documents/chunks out."""

    def split_document(self, document: Document) -> list[Document]:
        ...

    def split_documents(self, documents: Iterable[Document]) -> list[Document]:
        ...

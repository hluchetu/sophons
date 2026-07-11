from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Protocol, runtime_checkable

from sophons.documents import Document


@runtime_checkable
class DocumentStore(Protocol):
    """Document storage contract."""

    def put(self, document: Document) -> Document:
        ...

    def put_many(self, documents: Iterable[Document]) -> list[Document]:
        ...

    def get(self, document_id: str) -> Document | None:
        ...

    def list(
        self,
        *,
        filters: dict[str, Any] | None = None,
        limit: int | None = 100,
    ) -> list[Document]:
        ...

    def delete(self, document_id: str) -> bool:
        ...

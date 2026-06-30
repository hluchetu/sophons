from __future__ import annotations

from collections.abc import Iterable
from typing import Any
from uuid import uuid4

from sophons.documents import Document


class InMemoryDocumentStore:
    """Simple in-memory document store for tests and local prototypes."""

    def __init__(self, documents: Iterable[Document] | None = None) -> None:
        self._documents: dict[str, Document] = {}
        if documents is not None:
            self.put_many(documents)

    def put(self, document: Document) -> Document:
        stored = document if document.id is not None else document.with_id(self._new_id())
        if stored.id is None:
            raise ValueError("Stored document must have an ID.")
        self._documents[stored.id] = stored
        return stored

    def put_many(self, documents: Iterable[Document]) -> list[Document]:
        return [self.put(document) for document in documents]

    def get(self, document_id: str) -> Document | None:
        return self._documents.get(document_id)

    def list(
        self,
        *,
        filters: dict[str, Any] | None = None,
        limit: int | None = 100,
    ) -> list[Document]:
        documents = list(self._documents.values())
        if filters:
            documents = [
                document
                for document in documents
                if self._matches_filters(document, filters)
            ]
        if limit is not None:
            return documents[:limit]
        return documents

    def delete(self, document_id: str) -> bool:
        return self._documents.pop(document_id, None) is not None

    @staticmethod
    def _matches_filters(document: Document, filters: dict[str, Any]) -> bool:
        for key, value in filters.items():
            if key == "id":
                if document.id != value:
                    return False
                continue
            if document.metadata.get(key) != value:
                return False
        return True

    @staticmethod
    def _new_id() -> str:
        return f"doc_{uuid4().hex}"

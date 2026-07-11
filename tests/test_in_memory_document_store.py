from __future__ import annotations

from sophons.documents import Document
from sophons.storage import DocumentStore
from sophons.storage.in_memory import InMemoryDocumentStore


def test_in_memory_document_store_puts_and_gets_documents() -> None:
    store = InMemoryDocumentStore()

    stored = store.put(Document(id="doc_1", content="hello"))

    assert isinstance(store, DocumentStore)
    assert stored == Document(id="doc_1", content="hello")
    assert store.get("doc_1") == stored


def test_in_memory_document_store_assigns_missing_id() -> None:
    store = InMemoryDocumentStore()

    stored = store.put(Document(content="hello"))

    assert stored.id is not None
    assert stored.content == "hello"
    assert store.get(stored.id) == stored


def test_in_memory_document_store_lists_with_metadata_filters() -> None:
    store = InMemoryDocumentStore(
        [
            Document(id="cv", content="Python", metadata={"kind": "cv"}),
            Document(
                id="cover",
                content="Dear team",
                metadata={"kind": "cover_letter"},
            ),
        ]
    )

    assert store.list(filters={"kind": "cv"}) == [
        Document(id="cv", content="Python", metadata={"kind": "cv"})
    ]


def test_in_memory_document_store_deletes_documents() -> None:
    store = InMemoryDocumentStore([Document(id="doc_1", content="hello")])

    assert store.delete("doc_1") is True
    assert store.delete("doc_1") is False
    assert store.get("doc_1") is None

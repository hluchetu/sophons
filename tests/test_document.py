from __future__ import annotations

from sophons import Document


def test_document_defaults() -> None:
    document = Document(content="hello")

    assert document.content == "hello"
    assert document.metadata == {}
    assert document.id is None
    assert document.score is None


def test_document_copy_helpers() -> None:
    document = Document(content="hello", metadata={"source": "note"})

    updated = document.with_metadata(page=1).with_score(0.7).with_id("doc_1")

    assert document.metadata == {"source": "note"}
    assert document.score is None
    assert document.id is None
    assert updated.metadata == {"source": "note", "page": 1}
    assert updated.score == 0.7
    assert updated.id == "doc_1"


def test_document_round_trip_dict() -> None:
    document = Document(
        id="doc_1",
        content="hello",
        metadata={"source": "note"},
        score=0.7,
    )

    assert Document.from_dict(document.to_dict()) == document

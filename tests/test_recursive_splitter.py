from __future__ import annotations

import pytest

from sophons.documents import Document
from sophons.splitters import RecursiveCharacterSplitter


def test_recursive_splitter_keeps_small_document_as_one_chunk() -> None:
    splitter = RecursiveCharacterSplitter(chunk_size=100, chunk_overlap=0)
    document = Document(
        id="doc_1",
        content="Short CV summary.",
        metadata={"source": "cv.md"},
    )

    assert splitter.split_document(document) == [
        Document(
            id="doc_1#chunk_0",
            content="Short CV summary.",
            metadata={"source": "cv.md", "parent_id": "doc_1", "chunk_index": 0},
        )
    ]


def test_recursive_splitter_splits_by_paragraphs_first() -> None:
    splitter = RecursiveCharacterSplitter(chunk_size=30, chunk_overlap=0)
    document = Document(
        id="job",
        content="Backend engineer role.\n\nPython and RAG required.",
    )

    chunks = splitter.split_document(document)

    assert [chunk.content for chunk in chunks] == [
        "Backend engineer role.",
        "Python and RAG required.",
    ]
    assert [chunk.id for chunk in chunks] == ["job#chunk_0", "job#chunk_1"]


def test_recursive_splitter_adds_overlap() -> None:
    splitter = RecursiveCharacterSplitter(
        chunk_size=12,
        chunk_overlap=4,
        separators=[" "],
    )

    assert splitter.split_text("alpha beta gamma delta") == [
        "alpha beta",
        "beta gamma",
        "amma delta",
    ]


def test_recursive_splitter_keeps_chunk_id_none_without_parent_id() -> None:
    splitter = RecursiveCharacterSplitter(chunk_size=10, chunk_overlap=0)
    chunks = splitter.split_document(Document(content="hello world"))

    assert [chunk.id for chunk in chunks] == [None, None]


def test_recursive_splitter_rejects_invalid_overlap() -> None:
    with pytest.raises(ValueError, match="chunk_overlap must be smaller"):
        RecursiveCharacterSplitter(chunk_size=10, chunk_overlap=10)

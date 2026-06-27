from __future__ import annotations

from collections.abc import Iterable

from sophons.documents import Document
from sophons.splitters import Splitter


class LineSplitter:
    def split_document(self, document: Document) -> list[Document]:
        return [
            Document(
                id=f"{document.id or 'document'}#line_{index}",
                content=line,
                metadata={
                    **document.metadata,
                    "parent_id": document.id,
                    "chunk_index": index,
                },
            )
            for index, line in enumerate(document.content.splitlines())
            if line
        ]

    def split_documents(self, documents: Iterable[Document]) -> list[Document]:
        chunks: list[Document] = []
        for document in documents:
            chunks.extend(self.split_document(document))
        return chunks


def test_splitter_protocol_accepts_matching_object() -> None:
    splitter = LineSplitter()
    document = Document(
        id="doc_1",
        content="hello\nworld",
        metadata={"source": "note"},
    )

    assert isinstance(splitter, Splitter)
    assert splitter.split_documents([document]) == [
        Document(
            id="doc_1#line_0",
            content="hello",
            metadata={"source": "note", "parent_id": "doc_1", "chunk_index": 0},
        ),
        Document(
            id="doc_1#line_1",
            content="world",
            metadata={"source": "note", "parent_id": "doc_1", "chunk_index": 1},
        ),
    ]

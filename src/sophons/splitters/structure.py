from __future__ import annotations

from collections.abc import Iterable

from sophons.documents import Document
from sophons.parsers import DocumentBlock, StructureParser, TextStructureParser


class StructureAwareSplitter:
    """Pack intact blocks into chunks without crossing heading boundaries.

    max_chunk_size is a character target, not a hard limit: an oversized
    block is emitted intact and marked oversized rather than cutting a table,
    code block, paragraph, or clause. No overlap is added.
    """

    def __init__(self, *, parser: StructureParser | None = None,
                 max_chunk_size: int = 1000) -> None:
        if max_chunk_size <= 0:
            raise ValueError('max_chunk_size must be greater than 0')
        self.parser = parser or TextStructureParser()
        self.max_chunk_size = max_chunk_size

    def split(self, document: Document) -> list[Document]:
        """Compatibility with the RAG TextSplitter interface."""
        return self.split_document(document)

    def split_documents(self, documents: Iterable[Document]) -> list[Document]:
        return [chunk for document in documents for chunk in self.split_document(document)]

    def split_document(self, document: Document) -> list[Document]:
        chunks: list[Document] = []
        pending: list[DocumentBlock] = []

        def emit() -> None:
            if not pending:
                return
            content = '\n\n'.join(block.content for block in pending)
            index = len(chunks)
            page_metadata = {}
            if 'page_line_ranges' in document.metadata:
                pages = sorted({span['page'] for span in document.metadata['page_line_ranges']
                                if span['start_line'] <= pending[-1].end_line
                                and span['end_line'] >= pending[0].start_line})
                if pages:
                    page_metadata = {'pages': pages, 'page': pages[0],
                                     'start_page': pages[0], 'end_page': pages[-1]}
            chunks.append(Document(
                content=content,
                id=f'{document.id}#structure_{index}' if document.id is not None else None,
                metadata={**{key: value for key, value in document.metadata.items()
                              if key != 'page_line_ranges'},
                          **page_metadata, 'parent_id': document.id,
                          'chunk_index': index,
                          'heading_path': list(pending[0].heading_path),
                          'block_types': [block.kind for block in pending],
                          'start_line': pending[0].start_line,
                          'end_line': pending[-1].end_line,
                          'oversized': len(content) > self.max_chunk_size},
            ))
            pending.clear()

        for block in self.parser.parse(document):
            if not block.content.strip():
                continue
            size = sum(len(item.content) for item in pending) + 2 * len(pending) + len(block.content)
            if pending and (block.kind == 'heading' or
                            block.heading_path != pending[0].heading_path or
                            size > self.max_chunk_size):
                emit()
            pending.append(block)
        emit()
        return chunks

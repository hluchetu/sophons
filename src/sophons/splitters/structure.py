from __future__ import annotations

from collections.abc import Iterable

from sophons.documents import Document
from sophons.parsers import DocumentBlock, StructureParser, TextStructureParser
from sophons.parsers.elements import Element, ParsedDocument


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

    def split_parsed(self, parsed: ParsedDocument, *, id: str | None = None,
                     metadata: dict | None = None) -> list[Document]:
        """Pack a parsed document's elements into chunks without re-reading text.

        A chunk never crosses a heading boundary and never cuts an element; a
        table stays whole. Each chunk records its offsets into ``parsed.text``,
        so a quotation found in a chunk can be located in the source.
        """
        parent_id = id if id is not None else parsed.source
        base = {key: value for key, value in {
            'source': parsed.source, 'mime_type': parsed.mime_type, 'parser': parsed.parser,
            **parsed.metadata, **(metadata or {}),
        }.items() if value is not None}
        paths = parsed.heading_paths()
        chunks: list[Document] = []
        pending: list[Element] = []

        def emit() -> None:
            if not pending:
                return
            previous_id = None
            # A sentence cut by a page break is rejoined; other elements stay apart.
            content = ''
            for element in pending:
                body = parsed.text_of(element).strip()
                joined = bool(content) and element.metadata.get('continued_from') == previous_id
                content += (' ' if joined else '\n\n' if content else '') + body
                previous_id = element.id
            index = len(chunks)
            start, end = pending[0].start, pending[-1].end
            pages = parsed.page_indexes(start, end)
            page_metadata = {}
            if pages:
                numbers = [page + 1 for page in pages]
                page_metadata = {'pages': numbers, 'page': numbers[0],
                                 'start_page': numbers[0], 'end_page': numbers[-1]}
            flags = sorted({flag for element in pending for flag in element.quality_flags})
            chunks.append(Document(
                content=content,
                id=f'{parent_id}#structure_{index}' if parent_id is not None else None,
                metadata={**base, **page_metadata, 'parent_id': parent_id,
                          'chunk_index': index,
                          'heading_path': list(paths[pending[0].id]),
                          'block_types': [element.kind for element in pending],
                          'element_ids': [element.id for element in pending],
                          'start': start, 'end': end,
                          'methods': sorted({element.method for element in pending}),
                          **({'quality_flags': flags} if flags else {}),
                          'oversized': len(content) > self.max_chunk_size},
            ))
            pending.clear()

        for element in parsed.leaves():
            body = parsed.text_of(element)
            if not body.strip():
                continue
            size = sum(e.end - e.start for e in pending) + 2 * len(pending) + len(body)
            continues = bool(pending) and element.metadata.get('continued_from') == pending[-1].id
            if pending and not continues and (element.kind == 'heading' or
                                              paths[element.id] != paths[pending[0].id] or
                                              size > self.max_chunk_size):
                emit()
            pending.append(element)
        emit()
        return chunks

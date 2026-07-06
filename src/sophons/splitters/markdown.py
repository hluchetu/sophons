from __future__ import annotations

from collections.abc import Iterable

from sophons.documents import Document
from sophons.observability import SpanKind, Tracer, maybe_span


class MarkdownSplitter:
    """Split Markdown documents by headings."""

    def __init__(
        self,
        *,
        max_heading_level: int = 6,
        tracer: Tracer | None = None,
    ) -> None:
        if max_heading_level < 1 or max_heading_level > 6:
            raise ValueError("max_heading_level must be between 1 and 6.")
        self.max_heading_level = max_heading_level
        self._tracer = tracer

    def split_document(self, document: Document) -> list[Document]:
        sections = self.split_text(document.content)
        return [
            Document(
                id=self._chunk_id(document, index),
                content=section.content,
                metadata={
                    **document.metadata,
                    "parent_id": document.id,
                    "chunk_index": index,
                    "heading_path": section.heading_path,
                },
            )
            for index, section in enumerate(sections)
        ]

    def split_documents(self, documents: Iterable[Document]) -> list[Document]:
        with maybe_span(
            self._tracer,
            "splitter.split",
            kind=SpanKind.SPLITTER,
            splitter="markdown",
        ) as span:
            document_count = 0
            chunks: list[Document] = []
            for document in documents:
                document_count += 1
                chunks.extend(self.split_document(document))
            span.set_attribute("document_count", document_count)
            span.set_attribute("chunk_count", len(chunks))
            return chunks

    def split_text(self, text: str) -> list["_MarkdownSection"]:
        sections: list[_MarkdownSection] = []
        heading_stack: list[tuple[int, str]] = []
        current_lines: list[str] = []
        current_heading_path: list[str] = []

        for line in text.splitlines():
            heading = self._parse_heading(line)
            if heading is not None:
                if current_lines:
                    sections.append(
                        _MarkdownSection(
                            content="\n".join(current_lines).strip(),
                            heading_path=list(current_heading_path),
                        )
                    )
                    current_lines = []

                level, title = heading
                heading_stack = [
                    item for item in heading_stack if item[0] < level
                ]
                heading_stack.append((level, title))
                current_heading_path = [item[1] for item in heading_stack]
                current_lines.append(line)
                continue

            current_lines.append(line)

        if current_lines:
            sections.append(
                _MarkdownSection(
                    content="\n".join(current_lines).strip(),
                    heading_path=list(current_heading_path),
                )
            )

        return [section for section in sections if section.content]

    def _parse_heading(self, line: str) -> tuple[int, str] | None:
        stripped = line.lstrip()
        if not stripped.startswith("#"):
            return None

        marker = stripped.split(" ", 1)[0]
        if not marker or any(char != "#" for char in marker):
            return None

        level = len(marker)
        if level > self.max_heading_level or level == len(stripped):
            return None

        title = stripped[level:].strip()
        if not title:
            return None

        return level, title

    @staticmethod
    def _chunk_id(document: Document, index: int) -> str | None:
        if document.id is None:
            return None
        return f"{document.id}#section_{index}"


class _MarkdownSection:
    def __init__(self, *, content: str, heading_path: list[str]) -> None:
        self.content = content
        self.heading_path = heading_path

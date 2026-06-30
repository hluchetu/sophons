from __future__ import annotations

from collections.abc import Iterable
from html.parser import HTMLParser

from sophons.documents import Document


class HTMLSplitter:
    """Split HTML documents into text chunks using common block tags."""

    def split_document(self, document: Document) -> list[Document]:
        chunks = self.split_text(document.content)
        return [
            Document(
                id=self._chunk_id(document, index),
                content=chunk.content,
                metadata={
                    **document.metadata,
                    "parent_id": document.id,
                    "chunk_index": index,
                    "tag": chunk.tag,
                    "heading_path": chunk.heading_path,
                },
            )
            for index, chunk in enumerate(chunks)
        ]

    def split_documents(self, documents: Iterable[Document]) -> list[Document]:
        chunks: list[Document] = []
        for document in documents:
            chunks.extend(self.split_document(document))
        return chunks

    def split_text(self, html: str) -> list["_HTMLChunk"]:
        parser = _HTMLChunkParser()
        parser.feed(html)
        return parser.chunks()

    @staticmethod
    def _chunk_id(document: Document, index: int) -> str | None:
        if document.id is None:
            return None
        return f"{document.id}#html_{index}"


class _HTMLChunkParser(HTMLParser):
    _heading_tags = {"h1", "h2", "h3", "h4", "h5", "h6"}
    _chunk_tags = {"p", "li", "section", "article", "div", *_heading_tags}
    _skip_tags = {"script", "style", "noscript"}

    def __init__(self) -> None:
        super().__init__()
        self._skip_depth = 0
        self._tag_stack: list[str] = []
        self._text_stack: list[list[str]] = []
        self._chunks: list[_HTMLChunk] = []
        self._heading_stack: list[tuple[int, str]] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        tag = tag.lower()
        if tag in self._skip_tags:
            self._skip_depth += 1
            return

        self._tag_stack.append(tag)
        self._text_stack.append([])

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in self._skip_tags:
            if self._skip_depth:
                self._skip_depth -= 1
            return

        if not self._tag_stack:
            return

        open_tag = self._tag_stack.pop()
        text_parts = self._text_stack.pop()
        text = " ".join(" ".join(text_parts).split())

        if self._text_stack and text:
            self._text_stack[-1].append(text)

        if open_tag not in self._chunk_tags or not text:
            return

        heading_path = self._current_heading_path()
        if open_tag in self._heading_tags:
            level = int(open_tag[1])
            self._heading_stack = [
                item for item in self._heading_stack if item[0] < level
            ]
            self._heading_stack.append((level, text))
            heading_path = self._current_heading_path()

        self._chunks.append(
            _HTMLChunk(
                content=text,
                tag=open_tag,
                heading_path=heading_path,
            )
        )

    def handle_data(self, data: str) -> None:
        if self._skip_depth or not self._text_stack:
            return

        text = " ".join(data.split())
        if text:
            self._text_stack[-1].append(text)

    def chunks(self) -> list["_HTMLChunk"]:
        return self._dedupe_nested_chunks(self._chunks)

    def _current_heading_path(self) -> list[str]:
        return [heading for _, heading in self._heading_stack]

    @staticmethod
    def _dedupe_nested_chunks(chunks: list["_HTMLChunk"]) -> list["_HTMLChunk"]:
        deduped: list[_HTMLChunk] = []
        seen: set[tuple[str, str, tuple[str, ...]]] = set()

        for chunk in chunks:
            key = (chunk.content, chunk.tag, tuple(chunk.heading_path))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(chunk)

        return deduped


class _HTMLChunk:
    def __init__(self, *, content: str, tag: str, heading_path: list[str]) -> None:
        self.content = content
        self.tag = tag
        self.heading_path = heading_path

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sophons.documents import Document

# Kinds a parser may emit. Containers hold other elements; the rest are leaves.
CONTAINER_KINDS = frozenset({"page", "table", "row"})
ELEMENT_KINDS = CONTAINER_KINDS | frozenset(
    {"heading", "paragraph", "list_item", "cell", "code", "header", "footer", "caption", "footnote"}
)


class InvalidParsedDocument(ValueError):
    """A parser produced elements that do not describe the text they came from."""


@dataclass(frozen=True, slots=True)
class Element:
    """A typed span over ``ParsedDocument.text``. It carries no text of its own."""

    id: str
    kind: str
    start: int
    end: int
    parent_id: str | None = None
    level: int | None = None  # heading depth, 1 = top
    page_index: int | None = None  # 0-based
    page_label: str | None = None  # the label printed on the page, if known
    row_index: int | None = None
    column_index: int | None = None
    method: str = "text"  # how it was read: text, pdf, docx, ocr, html, layout
    quality_flags: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "start": self.start,
            "end": self.end,
            "parent_id": self.parent_id,
            "level": self.level,
            "page_index": self.page_index,
            "page_label": self.page_label,
            "row_index": self.row_index,
            "column_index": self.column_index,
            "method": self.method,
            "quality_flags": list(self.quality_flags),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Element:
        return cls(
            id=data["id"],
            kind=data["kind"],
            start=data["start"],
            end=data["end"],
            parent_id=data.get("parent_id"),
            level=data.get("level"),
            page_index=data.get("page_index"),
            page_label=data.get("page_label"),
            row_index=data.get("row_index"),
            column_index=data.get("column_index"),
            method=data.get("method", "text"),
            quality_flags=tuple(data.get("quality_flags") or ()),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass(frozen=True, slots=True)
class ParsedDocument:
    """Extracted text plus the typed elements that describe its structure.

    ``text`` is the single source of truth: every element, chunk and citation is
    a pair of offsets into it. Offsets never refer to the original file's bytes.
    """

    text: str
    elements: tuple[Element, ...]
    mime_type: str
    parser: str
    source: str | None = None
    checksum: str | None = None
    quality_flags: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "elements", tuple(self.elements))
        object.__setattr__(self, "quality_flags", tuple(self.quality_flags))

    # --- integrity -----------------------------------------------------------

    def validate(self) -> ParsedDocument:
        """Raise ``InvalidParsedDocument`` unless the elements are consistent."""

        by_id: dict[str, Element] = {}
        previous_start = 0
        for element in self.elements:
            if element.id in by_id:
                raise InvalidParsedDocument(f"Duplicate element ID {element.id!r}")
            if element.kind not in ELEMENT_KINDS:
                raise InvalidParsedDocument(f"Unknown element kind {element.kind!r}")
            if not 0 <= element.start <= element.end <= len(self.text):
                raise InvalidParsedDocument(f"Element {element.id!r} lies outside the text")
            if element.start < previous_start:
                raise InvalidParsedDocument(f"Element {element.id!r} is out of reading order")
            previous_start = element.start
            if element.parent_id is not None:
                parent = by_id.get(element.parent_id)
                if parent is None:
                    # Parents come first, so an unknown parent is missing or later.
                    raise InvalidParsedDocument(
                        f"Element {element.id!r} has no earlier parent {element.parent_id!r}"
                    )
                if not parent.start <= element.start <= element.end <= parent.end:
                    raise InvalidParsedDocument(f"Element {element.id!r} lies outside its parent")
            if element.level is not None and element.level < 1:
                raise InvalidParsedDocument(f"Element {element.id!r} has an invalid level")
            by_id[element.id] = element
        return self

    # --- reading -------------------------------------------------------------

    def text_of(self, element: Element) -> str:
        return self.text[element.start : element.end]

    def children(self, element_id: str) -> list[Element]:
        return [element for element in self.elements if element.parent_id == element_id]

    def leaves(self) -> list[Element]:
        """Content-bearing units for chunking, in reading order.

        A table is one unit (its rows and cells stay available for lookup), and
        pages are containers, not content.
        """

        tables = {element.id for element in self.elements if element.kind == "table"}
        rows = {
            element.id
            for element in self.elements
            if element.kind == "row" and element.parent_id in tables
        }
        units = []
        for element in self.elements:
            if element.kind == "page" or element.kind == "row":
                continue
            if element.kind == "cell" and element.parent_id in rows:
                continue
            units.append(element)
        return units

    def heading_paths(self) -> dict[str, tuple[str, ...]]:
        """The enclosing headings of each element, outermost first."""

        paths: dict[str, tuple[str, ...]] = {}
        stack: list[tuple[int, str]] = []
        for element in self.elements:
            if element.kind == "heading":
                level = element.level or 1
                stack = [(depth, title) for depth, title in stack if depth < level]
                stack.append((level, self.text_of(element).strip()))
            paths[element.id] = tuple(title for _, title in stack)
        return paths

    def page_indexes(self, start: int, end: int) -> list[int]:
        """Pages overlapped by a span, from page elements or element page numbers."""

        pages = {
            element.page_index
            for element in self.elements
            if element.page_index is not None and element.start < max(end, start + 1) and element.end > start
        }
        return sorted(pages)

    # --- conversion ----------------------------------------------------------

    def to_document(self, *, id: str | None = None, metadata: dict[str, Any] | None = None) -> Document:
        """Flatten to a plain ``Document`` for callers that do not need structure."""

        page_count = len({e.page_index for e in self.elements if e.kind == "page"})
        merged = {
            "source": self.source,
            "mime_type": self.mime_type,
            "parser": self.parser,
            **({"total_pages": page_count} if page_count else {}),
            **({"quality_flags": list(self.quality_flags)} if self.quality_flags else {}),
            **self.metadata,
            **(metadata or {}),
        }
        return Document(
            id=id if id is not None else self.source,
            content=self.text,
            metadata={key: value for key, value in merged.items() if value is not None},
        )

    def to_markdown(self) -> str:
        """Render headings, lists, code and tables as Markdown."""

        parts: list[str] = []
        for element in self.leaves():
            body = self.text_of(element).strip()
            if not body:
                continue
            if element.kind == "heading":
                parts.append("#" * min(element.level or 1, 6) + " " + " ".join(body.split()))
            elif element.kind == "table":
                rows = [
                    [" ".join(self.text_of(cell).split()) for cell in self.children(row.id)]
                    for row in self.children(element.id)
                    if row.kind == "row"
                ]
                rows = [row for row in rows if row]
                if not rows:
                    parts.append(body)
                    continue
                width = max(len(row) for row in rows)
                lines = ["| " + " | ".join(row + [""] * (width - len(row))) + " |" for row in rows]
                lines.insert(1, "| " + " | ".join(["---"] * width) + " |")
                parts.append("\n".join(lines))
            else:
                parts.append(body)
        return "\n\n".join(parts)

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe form, for crossing a process boundary or storing an artifact."""

        return {
            "text": self.text,
            "elements": [element.to_dict() for element in self.elements],
            "mime_type": self.mime_type,
            "parser": self.parser,
            "source": self.source,
            "checksum": self.checksum,
            "quality_flags": list(self.quality_flags),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ParsedDocument:
        return cls(
            text=data["text"],
            elements=tuple(Element.from_dict(item) for item in data["elements"]),
            mime_type=data["mime_type"],
            parser=data["parser"],
            source=data.get("source"),
            checksum=data.get("checksum"),
            quality_flags=tuple(data.get("quality_flags") or ()),
            metadata=dict(data.get("metadata") or {}),
        ).validate()

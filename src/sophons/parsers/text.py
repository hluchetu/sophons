from __future__ import annotations

import re

from sophons.errors import LoaderError
from sophons.parsers.blob import Blob
from sophons.parsers.elements import Element, ParsedDocument

_PARAGRAPH = re.compile(r"\S[\s\S]*?(?=\n[ \t]*\n|\Z)")
_FENCE = re.compile(r"^\s*(`{3,}|~{3,})")
_HEADING = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$")
_LIST_ITEM = re.compile(r"^\s*(?:[-+*]|\d+[.)])\s+")
_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")
_TABLE_RULE = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$")


def decode(blob: Blob) -> str:
    try:
        return blob.data.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise LoaderError("Text input is not valid UTF-8") from error


def paragraphs(text: str, *, offset: int = 0) -> list[tuple[int, int]]:
    """Spans of blank-line-separated paragraphs, shifted by ``offset``."""

    spans = []
    for match in _PARAGRAPH.finditer(text):
        end = match.start() + len(match.group().rstrip())
        spans.append((offset + match.start(), offset + end))
    return spans


class PlainTextParser:
    """Paragraphs split on blank lines. Makes no claim about headings."""

    name = "plain-text"

    def parse(self, blob: Blob) -> ParsedDocument:
        text = decode(blob)
        elements = [
            Element(
                id=f"paragraph-{index}",
                kind="paragraph",
                start=start,
                end=end,
                method="text",
                quality_flags=("paragraph_boundary_heuristic",),
            )
            for index, (start, end) in enumerate(paragraphs(text))
        ]
        return ParsedDocument(
            text=text,
            elements=tuple(elements),
            mime_type=blob.mime_type,
            parser=self.name,
            source=blob.source,
            checksum=blob.checksum,
            quality_flags=("structure_unresolved",),
            metadata=dict(blob.metadata),
        )


class MarkdownParser:
    """Headings, paragraphs, list items, fenced code and pipe tables.

    The text is kept exactly as written; elements are spans over it.
    """

    name = "markdown"

    def parse(self, blob: Blob) -> ParsedDocument:
        text = decode(blob)
        lines = text.splitlines(keepends=True)
        starts, position = [], 0
        for line in lines:
            starts.append(position)
            position += len(line)
        elements: list[Element] = []
        counts: dict[str, int] = {}

        def span(first: int, last: int) -> tuple[int, int]:
            """Offsets covering lines first..last, without the trailing newline."""
            end = starts[last] + len(lines[last].rstrip("\r\n"))
            return starts[first], end

        def add(kind: str, first: int, last: int, **fields) -> Element:
            index = counts.get(kind, 0)
            counts[kind] = index + 1
            start, end = span(first, last)
            element = Element(
                id=f"{kind}-{index}", kind=kind, start=start, end=end, method="text", **fields
            )
            elements.append(element)
            return element

        index = 0
        while index < len(lines):
            line = lines[index]
            if not line.strip():
                index += 1
                continue
            fence = _FENCE.match(line)
            if fence:
                marker, first = fence.group(1), index
                index += 1
                closing = re.compile(r"^\s*" + re.escape(marker[0]) + "{" + str(len(marker)) + r",}\s*$")
                while index < len(lines):
                    index += 1
                    if closing.match(lines[index - 1]):
                        break
                add("code", first, index - 1)
                continue
            heading = _HEADING.match(line)
            if heading:
                # The element is the title itself, without the leading # markers.
                count = counts.get("heading", 0)
                counts["heading"] = count + 1
                elements.append(
                    Element(
                        id=f"heading-{count}",
                        kind="heading",
                        start=starts[index] + heading.start(2),
                        end=starts[index] + heading.end(2),
                        level=len(heading.group(1)),
                        method="text",
                    )
                )
                index += 1
                continue
            if _TABLE_ROW.match(line):
                first = index
                while index < len(lines) and _TABLE_ROW.match(lines[index]):
                    index += 1
                table = add("table", first, index - 1)
                self._rows(table, lines, starts, first, index, elements)
                continue
            if _LIST_ITEM.match(line):
                first = index
                index += 1
                # Continuation lines belong to the item until the next item or a blank line.
                while (
                    index < len(lines)
                    and lines[index].strip()
                    and not _LIST_ITEM.match(lines[index])
                    and not _HEADING.match(lines[index])
                    and not _FENCE.match(lines[index])
                    and not _TABLE_ROW.match(lines[index])
                ):
                    index += 1
                add("list_item", first, index - 1)
                continue
            first = index
            index += 1
            while (
                index < len(lines)
                and lines[index].strip()
                and not _HEADING.match(lines[index])
                and not _FENCE.match(lines[index])
                and not _LIST_ITEM.match(lines[index])
                and not _TABLE_ROW.match(lines[index])
            ):
                index += 1
            add("paragraph", first, index - 1)
        return ParsedDocument(
            text=text,
            elements=tuple(elements),
            mime_type=blob.mime_type,
            parser=self.name,
            source=blob.source,
            checksum=blob.checksum,
            metadata=dict(blob.metadata),
        )

    @staticmethod
    def _rows(table: Element, lines, starts, first: int, stop: int, elements: list[Element]) -> None:
        row_index = 0
        for number in range(first, stop):
            line = lines[number].rstrip("\r\n")
            if _TABLE_RULE.match(line):
                continue
            row_id = f"{table.id}-row-{row_index}"
            elements.append(
                Element(
                    id=row_id,
                    kind="row",
                    start=starts[number] + (len(line) - len(line.lstrip())),
                    end=starts[number] + len(line.rstrip()),
                    parent_id=table.id,
                    row_index=row_index,
                    method="text",
                )
            )
            inner_start = line.index("|") + 1
            inner_end = line.rstrip().rindex("|")
            column, cursor = 0, inner_start
            for cell in line[inner_start:inner_end].split("|"):
                stripped = cell.strip()
                lead = len(cell) - len(cell.lstrip())
                elements.append(
                    Element(
                        id=f"{row_id}-cell-{column}",
                        kind="cell",
                        start=starts[number] + cursor + (lead if stripped else 0),
                        end=starts[number] + cursor + (lead + len(stripped) if stripped else 0),
                        parent_id=row_id,
                        row_index=row_index,
                        column_index=column,
                        method="text",
                    )
                )
                cursor += len(cell) + 1
                column += 1
            row_index += 1

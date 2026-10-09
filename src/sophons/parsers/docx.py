from __future__ import annotations

import re
from io import BytesIO
from xml.etree import ElementTree
from zipfile import BadZipFile, ZipFile

from sophons.errors import LoaderError
from sophons.parsers.blob import Blob
from sophons.parsers.elements import Element, ParsedDocument

_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
_MAX_ENTRIES, _MAX_EXPANDED = 5000, 100 * 1024 * 1024


class DocxParser:
    """Word documents read from their XML: no layout guessing is needed.

    Headings come only from Word heading styles; a bold line typed as a heading
    is an ordinary paragraph here. Tables keep their rows and cells. Merged cells
    are flagged, not reconstructed.
    """

    name = "docx"

    def parse(self, blob: Blob) -> ParsedDocument:
        try:
            with ZipFile(BytesIO(blob.data)) as archive:
                entries = archive.infolist()
                if len(entries) > _MAX_ENTRIES or sum(e.file_size for e in entries) > _MAX_EXPANDED:
                    raise LoaderError("DOCX expanded content exceeds processing limits")
                body = ElementTree.fromstring(archive.read("word/document.xml")).find(f"{_W}body")
                styles = self._heading_styles(archive)
        except (BadZipFile, KeyError, ElementTree.ParseError) as error:
            raise LoaderError("The file is not a readable DOCX document") from error

        parts: list[str] = []
        elements: list[Element] = []
        offset = 0
        for index, node in enumerate(self._blocks(body) if body is not None else ()):
            block_id = f"docx-{index}"
            start = offset + (2 if parts else 0)
            if node.tag == f"{_W}p":
                text = self._paragraph_text(node)
                if not text.strip():
                    continue
                level = self._heading_level(node, styles)
                kind = "heading" if level else "list_item" if node.find(f"{_W}pPr/{_W}numPr") is not None else "paragraph"
                elements.append(
                    Element(id=block_id, kind=kind, start=start, end=start + len(text), level=level, method="docx")
                )
            else:
                text, table = self._table(node, block_id, start)
                if not text.strip():
                    continue
                elements.extend(table)
            parts.append(text)
            offset = start + len(text)
        return ParsedDocument(
            text="\n\n".join(parts),
            elements=tuple(elements),
            mime_type=blob.mime_type,
            parser=self.name,
            source=blob.source,
            checksum=blob.checksum,
            metadata=dict(blob.metadata),
        )

    @classmethod
    def _blocks(cls, parent):
        """Paragraphs and tables in order, looking inside content controls."""
        for node in parent:
            if node.tag in (f"{_W}p", f"{_W}tbl"):
                yield node
            elif node.tag == f"{_W}sdt":
                content = node.find(f"{_W}sdtContent")
                if content is not None:
                    yield from cls._blocks(content)

    @staticmethod
    def _paragraph_text(paragraph) -> str:
        pieces = []
        for node in paragraph.iter():
            if node.tag == f"{_W}t":
                pieces.append(node.text or "")
            elif node.tag == f"{_W}tab":
                pieces.append("\t")
            elif node.tag in (f"{_W}br", f"{_W}cr"):
                pieces.append("\n")
        return "".join(pieces).strip()

    @staticmethod
    def _heading_styles(archive: ZipFile) -> dict[str, int]:
        """Style ID to heading level, from the style names Word records."""
        levels: dict[str, int] = {}
        try:
            root = ElementTree.fromstring(archive.read("word/styles.xml"))
        except (KeyError, ElementTree.ParseError):
            return levels
        for style in root.iter(f"{_W}style"):
            name = style.find(f"{_W}name")
            label = (name.get(f"{_W}val") if name is not None else "") or ""
            match = re.fullmatch(r"heading\s*(\d)", label.strip().lower())
            if match:
                levels[style.get(f"{_W}styleId", "")] = int(match.group(1))
            elif label.strip().lower() == "title":
                levels[style.get(f"{_W}styleId", "")] = 1
        return levels

    @staticmethod
    def _heading_level(paragraph, styles: dict[str, int]) -> int | None:
        style = paragraph.find(f"{_W}pPr/{_W}pStyle")
        if style is None:
            return None
        style_id = style.get(f"{_W}val", "")
        if style_id in styles:
            return styles[style_id]
        match = re.fullmatch(r"heading\s*(\d)", style_id.strip().lower())
        return int(match.group(1)) if match else None

    def _table(self, table, block_id: str, start: int) -> tuple[str, list[Element]]:
        rows_text: list[str] = []
        children: list[Element] = []
        merged = False
        offset = start
        for row_index, row in enumerate(table.findall(f"{_W}tr")):
            if rows_text:
                offset += 1
            row_start = offset
            row_id = f"{block_id}-row-{row_index}"
            cells_text: list[str] = []
            cells: list[Element] = []
            for column, cell in enumerate(row.findall(f"{_W}tc")):
                if cells_text:
                    offset += 1
                properties = cell.find(f"{_W}tcPr")
                if properties is not None and (
                    properties.find(f"{_W}gridSpan") is not None or properties.find(f"{_W}vMerge") is not None
                ):
                    merged = True
                text = " ".join(
                    part for part in (self._paragraph_text(p) for p in cell.iter(f"{_W}p")) if part
                )
                cells.append(
                    Element(
                        id=f"{row_id}-cell-{column}",
                        kind="cell",
                        start=offset,
                        end=offset + len(text),
                        parent_id=row_id,
                        row_index=row_index,
                        column_index=column,
                        method="docx",
                    )
                )
                cells_text.append(text)
                offset += len(text)
            children.append(
                Element(id=row_id, kind="row", start=row_start, end=offset, parent_id=block_id,
                        row_index=row_index, method="docx")
            )
            children.extend(cells)
            rows_text.append("\t".join(cells_text))
        text = "\n".join(rows_text)
        head = Element(
            id=block_id,
            kind="table",
            start=start,
            end=start + len(text),
            method="docx",
            quality_flags=("merged_cells_unresolved",) if merged else (),
        )
        return text, [head, *children]

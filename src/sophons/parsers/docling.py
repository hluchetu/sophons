from __future__ import annotations

import mimetypes
from io import BytesIO
from pathlib import Path
from typing import Any

from sophons.errors import LoaderError, MissingDependencyError
from sophons.parsers.blob import Blob
from sophons.parsers.elements import Element, ParsedDocument

# Docling item labels to Sophons element kinds. Anything unlisted that carries
# text is kept as a paragraph, so no recognized content is dropped.
_KINDS = {
    "title": "heading",
    "section_header": "heading",
    "text": "paragraph",
    "paragraph": "paragraph",
    "list_item": "list_item",
    "code": "code",
    "caption": "caption",
    "footnote": "footnote",
    "page_header": "header",
    "page_footer": "footer",
    "table": "table",
    "document_index": "contents",
}
_SUFFIXES = {
    "application/pdf": ".pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": ".pptx",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
    "text/html": ".html",
    "text/markdown": ".md",
}


class DoclingParser:
    """Layout-aware parsing with Docling: headings, lists, tables and furniture.

    Docling runs models locally to find each region of a page and its reading
    order, which a text-layer extractor cannot do. Every element is recorded as
    read by ``layout``, with its page and, when known, its position on the page.

    Requires the optional ``docling`` extra, which is large. Models are fetched
    on first use unless ``artifacts_path`` points at a local copy, which a
    process without network access needs. OCR is off by default: pass
    ``ocr=True`` once an OCR engine Docling supports is installed.
    """

    name = "docling"

    def __init__(
        self,
        *,
        ocr: bool = False,
        tables: bool = True,
        artifacts_path: str | Path | None = None,
        max_pages: int = 500,
        converter: Any | None = None,
    ) -> None:
        self.ocr = ocr
        self.tables = tables
        self.artifacts_path = artifacts_path
        self.max_pages = max_pages
        self._converter = converter

    def _load(self) -> Any:
        if self._converter is None:
            try:
                from docling.datamodel.base_models import InputFormat
                from docling.datamodel.pipeline_options import PdfPipelineOptions
                from docling.document_converter import DocumentConverter, PdfFormatOption
            except ImportError as exc:
                raise MissingDependencyError(
                    "DoclingParser requires docling. Install it with `pip install 'sophons[docling]'`.",
                    details={"dependency": "docling", "extra": "docling"},
                ) from exc
            options = PdfPipelineOptions(do_ocr=self.ocr, do_table_structure=self.tables)
            if self.artifacts_path is not None:
                options.artifacts_path = str(self.artifacts_path)
            self._converter = DocumentConverter(
                format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
            )
        return self._converter

    def parse(self, blob: Blob) -> ParsedDocument:
        if not blob.data:
            raise LoaderError("The document is empty")
        converter = self._load()
        try:
            result = converter.convert(self._source(blob), max_num_pages=self.max_pages)
        except Exception as exc:  # Docling raises several unrelated error types.
            raise LoaderError("Docling could not convert this document") from exc
        return self._build(blob, result.document)

    @staticmethod
    def _source(blob: Blob) -> Any:
        """A named in-memory stream; Docling picks the format from the suffix."""
        from docling.datamodel.base_models import DocumentStream

        name = Path(blob.source).name if blob.source else ""
        if not Path(name).suffix:
            suffix = _SUFFIXES.get(blob.mime_type) or mimetypes.guess_extension(blob.mime_type) or ""
            name = (name or "document") + suffix
        return DocumentStream(name=name, stream=BytesIO(blob.data))

    def _build(self, blob: Blob, document: Any) -> ParsedDocument:
        parts: list[str] = []
        elements: list[Element] = []
        offset = 0
        for index, item in enumerate(self._items(document)):
            label = _label(item)
            kind = _KINDS.get(label)
            page_index, bbox = _provenance(item)
            metadata = {"docling_label": label, **({"bbox": bbox} if bbox else {})}
            start = offset + (2 if parts else 0)
            element_id = f"docling-{index}"
            if label in ("table", "document_index"):
                body, children = _table(item, element_id, start, page_index)
                if not body.strip():
                    continue
                elements.append(
                    Element(
                        id=element_id,
                        kind=kind,
                        start=start,
                        end=start + len(body),
                        page_index=page_index,
                        method="layout",
                        quality_flags=("merged_cells_unresolved",) if children[1] else (),
                        metadata=metadata,
                    )
                )
                elements.extend(children[0])
            else:
                body = (getattr(item, "text", "") or "").strip()
                if not body:
                    continue
                level = None
                if label == "title":
                    level = 1
                elif label == "section_header":
                    level = max(int(getattr(item, "level", 1) or 1), 1)
                elements.append(
                    Element(
                        id=element_id,
                        kind=kind or "paragraph",
                        start=start,
                        end=start + len(body),
                        level=level,
                        page_index=page_index,
                        method="layout",
                        metadata=metadata,
                    )
                )
            parts.append(body)
            offset = start + len(body)
        return ParsedDocument(
            text="\n\n".join(parts),
            elements=tuple(elements),
            mime_type=blob.mime_type,
            parser=self.name,
            source=blob.source,
            checksum=blob.checksum,
            metadata=dict(blob.metadata),
        )

    @staticmethod
    def _items(document: Any):
        """Items in reading order, including page headers and footers."""
        try:
            from docling_core.types.doc import ContentLayer

            layers = {ContentLayer.BODY, ContentLayer.FURNITURE}
            iterator = document.iterate_items(with_groups=False, included_content_layers=layers)
        except ImportError:
            iterator = document.iterate_items(with_groups=False)
        for item, _depth in iterator:
            yield item


def _label(item: Any) -> str:
    label = getattr(item, "label", "")
    return str(getattr(label, "value", label)).lower()


def _provenance(item: Any) -> tuple[int | None, list[float] | None]:
    provenance = getattr(item, "prov", None) or []
    if not provenance:
        return None, None
    first = provenance[0]
    page = getattr(first, "page_no", None)
    box = getattr(first, "bbox", None)
    bbox = None
    if box is not None:
        bbox = [round(float(getattr(box, side)), 2) for side in ("l", "t", "r", "b")]
    return (page - 1 if isinstance(page, int) and page >= 1 else None), bbox


def _table(item: Any, element_id: str, start: int, page_index: int | None):
    """Rows and cells from Docling's table cells; a spanning cell appears once."""
    data = getattr(item, "data", None)
    cells = sorted(
        getattr(data, "table_cells", None) or [],
        key=lambda cell: (cell.start_row_offset_idx, cell.start_col_offset_idx),
    )
    merged = any((cell.row_span or 1) > 1 or (cell.col_span or 1) > 1 for cell in cells)
    rows: dict[int, list[Any]] = {}
    for cell in cells:
        rows.setdefault(cell.start_row_offset_idx, []).append(cell)
    children: list[Element] = []
    rows_text: list[str] = []
    offset = start
    for row_index in sorted(rows):
        if rows_text:
            offset += 1
        row_start = offset
        row_id = f"{element_id}-row-{row_index}"
        texts: list[str] = []
        row_cells: list[Element] = []
        for cell in rows[row_index]:
            if texts:
                offset += 1
            text = " ".join((cell.text or "").split())
            row_cells.append(
                Element(
                    id=f"{row_id}-cell-{cell.start_col_offset_idx}",
                    kind="cell",
                    start=offset,
                    end=offset + len(text),
                    parent_id=row_id,
                    page_index=page_index,
                    row_index=row_index,
                    column_index=cell.start_col_offset_idx,
                    method="layout",
                    metadata={"header": True} if getattr(cell, "column_header", False) else {},
                )
            )
            texts.append(text)
            offset += len(text)
        children.append(
            Element(
                id=row_id, kind="row", start=row_start, end=offset, parent_id=element_id,
                page_index=page_index, row_index=row_index, method="layout",
            )
        )
        children.extend(row_cells)
        rows_text.append("\t".join(texts))
    return "\n".join(rows_text), (children, merged)

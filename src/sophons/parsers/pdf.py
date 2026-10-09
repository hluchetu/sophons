from __future__ import annotations

from io import BytesIO

from sophons.errors import LoaderError, MissingDependencyError
from sophons.parsers.base import ParserDeclined
from sophons.parsers.blob import Blob
from sophons.parsers.elements import Element, ParsedDocument
from sophons.parsers.text import paragraphs


def page_document(
    blob: Blob,
    pages: list[tuple[str, str]],
    *,
    parser: str,
    labels: list[str] | None = None,
) -> ParsedDocument:
    """Build a document from per-page text and the method each page was read by.

    Empty pages are kept so page numbers stay true. Paragraph boundaries inside a
    page are a blank-line heuristic and flagged as such; layout is not resolved.
    """

    elements: list[Element] = []
    offset = 0
    for index, (body, method) in enumerate(pages):
        flags = ["layout_unresolved"]
        if method == "ocr":
            flags.insert(0, "ocr_unreviewed")
        if not body.strip():
            flags.append("empty_page")
        label = labels[index] if labels else None
        elements.append(
            Element(
                id=f"page-{index}",
                kind="page",
                start=offset,
                end=offset + len(body),
                page_index=index,
                page_label=label,
                method=method,
                quality_flags=tuple(flags),
            )
        )
        for number, (start, end) in enumerate(paragraphs(body, offset=offset)):
            elements.append(
                Element(
                    id=f"page-{index}-paragraph-{number}",
                    kind="paragraph",
                    start=start,
                    end=end,
                    parent_id=f"page-{index}",
                    page_index=index,
                    page_label=label,
                    method=method,
                    quality_flags=("paragraph_boundary_heuristic",)
                    + (("ocr_unreviewed",) if method == "ocr" else ()),
                )
            )
        offset += len(body) + 2
    return ParsedDocument(
        text="\n\n".join(body for body, _ in pages),
        elements=tuple(elements),
        mime_type=blob.mime_type,
        parser=parser,
        source=blob.source,
        checksum=blob.checksum,
        quality_flags=("layout_unresolved",),
        metadata=dict(blob.metadata),
    )


class PyPDFParser:
    """Text-layer extraction with pypdf: pages and paragraphs, no layout analysis.

    Declines a PDF that has a page without extractable text, so a registry can
    fall through to OCR instead of silently dropping that page.
    """

    name = "pypdf"

    def __init__(self, *, max_pages: int = 500, allow_empty_pages: bool = False) -> None:
        self.max_pages = max_pages
        self.allow_empty_pages = allow_empty_pages

    def parse(self, blob: Blob) -> ParsedDocument:
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise MissingDependencyError(
                "PyPDFParser requires pypdf. Install it with `pip install 'sophons[pdf]'`.",
                details={"dependency": "pypdf", "extra": "pdf"},
            ) from exc
        if not blob.data:
            raise LoaderError("The PDF is empty")
        reader = PdfReader(BytesIO(blob.data))
        if reader.is_encrypted:
            raise LoaderError("Password-protected PDFs are not supported")
        if len(reader.pages) > self.max_pages:
            raise LoaderError(f"PDFs must contain no more than {self.max_pages} pages")
        pages = [page.extract_text() or "" for page in reader.pages]
        if not self.allow_empty_pages and any(not text.strip() for text in pages):
            raise ParserDeclined("The PDF has pages without a text layer")
        labels = reader.page_labels if "/PageLabels" in reader.trailer["/Root"] else None
        return page_document(blob, [(text, "pdf") for text in pages], parser=self.name, labels=labels)

from __future__ import annotations

from pathlib import Path
from typing import Any

from sophons.documents import Document
from sophons.errors import MissingDependencyError


class PDFStructureLoader:
    """Convert PDF layout to Markdown using optional PyMuPDF4LLM.

    Returns one document so heading context can continue across pages.
    page_line_ranges maps generated Markdown lines back to source pages.
    OCR is opt-in and requires the backend's OCR dependencies.
    """

    def __init__(self, path: str | Path, *, id: str | None = None,
                 metadata: dict[str, Any] | None = None, use_ocr: bool = False) -> None:
        self.path = Path(path)
        self.id = id if id is not None else str(self.path)
        self.metadata = dict(metadata or {})
        self.use_ocr = use_ocr

    def load(self) -> list[Document]:
        try:
            import pymupdf4llm
        except ImportError as exc:
            raise MissingDependencyError(
                "PDFStructureLoader requires pymupdf4llm. "
                "Install it with `pip install 'sophons[pdf-structure]'`.",
                details={'dependency': 'pymupdf4llm', 'extra': 'pdf-structure'},
            ) from exc

        pages = pymupdf4llm.to_markdown(
            str(self.path), page_chunks=True, use_ocr=self.use_ocr,
            show_progress=False,
        )
        if not isinstance(pages, list):
            raise ValueError('PyMuPDF4LLM must return page chunks; received a non-list result.')
        lines: list[str] = []
        ranges: list[dict[str, int]] = []
        for index, page in enumerate(pages):
            page_lines = page['text'].strip().splitlines()
            if not page_lines:
                continue
            if lines:
                lines.append('')
            start = len(lines) + 1
            lines.extend(page_lines)
            ranges.append({
                'page': int(page.get('metadata', {}).get('page_number', index + 1)),
                'start_line': start, 'end_line': len(lines),
            })
        return [Document(
            id=self.id, content='\n'.join(lines),
            metadata={**self.metadata, 'source': str(self.path),
                      'file_name': self.path.name, 'mime_type': 'application/pdf',
                      'content_format': 'markdown', 'parser': 'pymupdf4llm',
                      'total_pages': len(pages), 'page_line_ranges': ranges},
        )]

    def lazy_load(self):
        yield from self.load()

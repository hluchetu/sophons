from __future__ import annotations

from collections.abc import Callable
from io import BytesIO
from typing import Any

from sophons.errors import LoaderError, MissingDependencyError
from sophons.parsers.base import ParserDeclined
from sophons.parsers.blob import Blob
from sophons.parsers.elements import ParsedDocument
from sophons.parsers.pdf import page_document

_INSTALL = "Install it with `pip install 'sophons[ocr]'`."


def _missing(dependency: str, error: Exception) -> MissingDependencyError:
    return MissingDependencyError(
        f"OCRParser requires {dependency}. {_INSTALL}",
        details={"dependency": dependency, "extra": "ocr"},
    )


class OCRParser:
    """Read scanned PDFs and images by recognizing text in page images.

    For a PDF, pages that already have a text layer keep it; only pages without
    one are rendered and recognized, and every page records which way it was
    read. OCR output is flagged ``ocr_unreviewed``: recognition can be wrong and
    is never presented as checked text.

    ``recognize`` turns one page image into text. It defaults to Tesseract; pass
    another callable to use a different engine.
    """

    name = "ocr"

    def __init__(
        self,
        *,
        language: str = "eng",
        max_pages: int = 500,
        max_page_pixels: int = 20_000_000,
        scale: float = 2.0,
        timeout_seconds: float = 60,
        recognize: Callable[[Any], str] | None = None,
    ) -> None:
        self.language = language
        self.max_pages = max_pages
        self.max_page_pixels = max_page_pixels
        self.scale = scale
        self.timeout_seconds = timeout_seconds
        self._recognize = recognize or self._tesseract

    def parse(self, blob: Blob) -> ParsedDocument:
        if not blob.data:
            raise LoaderError("The document is empty")
        labels = None
        if blob.mime_type.startswith("image/"):
            pages = [(self._image_text(blob.data), "ocr")]
        elif blob.mime_type == "application/pdf":
            pages, labels = self._pdf_pages(blob.data)
        else:
            raise ParserDeclined("OCR reads only PDFs and images")
        if not any(text.strip() for text, _ in pages):
            raise ParserDeclined("No readable text was found in the document")
        return page_document(blob, pages, parser=self.name, labels=labels)

    def _tesseract(self, image: Any) -> str:
        try:
            import pytesseract
        except ImportError as exc:
            raise _missing("pytesseract", exc) from exc
        try:
            return pytesseract.image_to_string(image, lang=self.language, timeout=self.timeout_seconds)
        except pytesseract.TesseractNotFoundError as exc:
            raise MissingDependencyError(
                "OCRParser requires the Tesseract program to be installed on this machine.",
                details={"dependency": "tesseract"},
            ) from exc

    def _image_text(self, data: bytes) -> str:
        try:
            from PIL import Image
        except ImportError as exc:
            raise _missing("pillow", exc) from exc
        with Image.open(BytesIO(data)) as image:
            width, height = image.size
            if width * height > self.max_page_pixels:
                raise LoaderError("The image exceeds the OCR pixel limit")
            return self._recognize(image.convert("RGB"))

    def _pdf_pages(self, data: bytes) -> tuple[list[tuple[str, str]], list[str] | None]:
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise _missing("pypdf", exc) from exc
        try:
            import pypdfium2 as pdfium
        except ImportError as exc:
            raise _missing("pypdfium2", exc) from exc
        reader = PdfReader(BytesIO(data))
        if reader.is_encrypted:
            raise LoaderError("Password-protected PDFs are not supported")
        if len(reader.pages) > self.max_pages:
            raise LoaderError(f"PDFs must contain no more than {self.max_pages} pages")
        labels = reader.page_labels if "/PageLabels" in reader.trailer["/Root"] else None
        pages: list[tuple[str, str]] = []
        document = pdfium.PdfDocument(data)
        try:
            for index, source in enumerate(reader.pages):
                text = source.extract_text() or ""
                if text.strip():
                    pages.append((text, "pdf"))
                    continue
                page = document[index]
                try:
                    width, height = page.get_size()
                    if width * self.scale * height * self.scale > self.max_page_pixels:
                        raise LoaderError("A PDF page exceeds the OCR pixel limit")
                    bitmap = page.render(scale=self.scale)
                    try:
                        pages.append((self._recognize(bitmap.to_pil()), "ocr"))
                    finally:
                        bitmap.close()
                finally:
                    page.close()
        finally:
            document.close()
        return pages, labels

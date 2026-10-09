from __future__ import annotations

from sophons.parsers.base import ParserRegistry
from sophons.parsers.docx import DocxParser
from sophons.parsers.ocr import OCRParser
from sophons.parsers.pdf import PyPDFParser
from sophons.parsers.text import MarkdownParser, PlainTextParser

DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


def default_registry() -> ParserRegistry:
    """Built-in parsers. Construction imports no optional dependency."""

    registry = ParserRegistry()
    registry.register("text/plain", PlainTextParser())
    registry.register("text/markdown", MarkdownParser())
    registry.register("application/pdf", PyPDFParser())
    # A PDF with pages lacking a text layer is declined above and read here.
    ocr = OCRParser()
    registry.register("application/pdf", ocr)
    registry.register("image/*", ocr)
    registry.register(DOCX, DocxParser())
    return registry

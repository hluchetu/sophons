from __future__ import annotations

from pathlib import Path
from typing import Any

from sophons.documents import Document
from sophons.errors import UnsupportedFileTypeError
from sophons.loaders.base import Loader
from sophons.loaders.docx import DocxLoader
from sophons.loaders.pdf import PDFLoader
from sophons.loaders.text import TextLoader
from sophons.observability import SpanKind, Tracer, maybe_span


class FileLoader:
    """Route a local file path to the right loader based on its suffix."""

    def __init__(
        self,
        path: str | Path,
        *,
        id: str | None = None,
        metadata: dict[str, Any] | None = None,
        tracer: Tracer | None = None,
    ) -> None:
        self.path = Path(path)
        self.id = id
        self.metadata = dict(metadata or {})
        self._tracer = tracer
        self._loader = self._select_loader()

    def load(self) -> list[Document]:
        with maybe_span(
            self._tracer,
            "loader.load",
            kind=SpanKind.LOADER,
            loader="file",
            path=str(self.path),
        ) as span:
            documents = self._loader.load()
            span.set_attribute("document_count", len(documents))
            return documents

    def lazy_load(self):
        yield from self._loader.lazy_load()

    def _select_loader(self) -> Loader:
        suffix = self.path.suffix.lower()

        if suffix in {".txt", ".md", ".markdown"}:
            return TextLoader.from_file(
                self.path,
                id=self.id,
                metadata={**self.metadata, "mime_type": self._text_mime_type(suffix)},
            )
        if suffix == ".docx":
            return DocxLoader(self.path, id=self.id, metadata=self.metadata)
        if suffix == ".pdf":
            return PDFLoader(self.path, id=self.id, metadata=self.metadata)

        raise UnsupportedFileTypeError(
            f"Unsupported file type {suffix!r} for {self.path}. "
            "Supported types: .txt, .md, .markdown, .docx, .pdf.",
            details={"path": str(self.path), "suffix": suffix},
        )

    @staticmethod
    def _text_mime_type(suffix: str) -> str:
        if suffix in {".md", ".markdown"}:
            return "text/markdown"
        return "text/plain"

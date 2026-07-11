from __future__ import annotations

from pathlib import Path
from typing import Any

from sophons.documents import Document
from sophons.errors import MissingDependencyError


class PDFLoader:
    """Load text from a PDF file into Sophons page documents."""

    def __init__(
        self,
        path: str | Path,
        *,
        id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.path = Path(path)
        self.id = id or str(self.path)
        self.metadata = dict(metadata or {})

    def load(self) -> list[Document]:
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise MissingDependencyError(
                "PDFLoader requires pypdf. Install it with `pip install 'sophons[pdf]'` "
                "or `pip install pypdf`.",
                details={"dependency": "pypdf", "extra": "pdf"},
            ) from exc

        reader = PdfReader(str(self.path))
        documents: list[Document] = []
        total_pages = len(reader.pages)

        for page_index, page in enumerate(reader.pages):
            metadata = {
                "source": str(self.path),
                "file_name": self.path.name,
                "mime_type": "application/pdf",
                "page": page_index + 1,
                "total_pages": total_pages,
            }
            metadata.update(self.metadata)

            documents.append(
                Document(
                    id=f"{self.id}#page_{page_index + 1}",
                    content=page.extract_text() or "",
                    metadata=metadata,
                )
            )

        return documents

    def lazy_load(self):
        yield from self.load()

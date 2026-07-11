from __future__ import annotations

from pathlib import Path
from typing import Any
from xml.etree import ElementTree
from zipfile import ZipFile

from sophons.documents import Document


class DocxLoader:
    """Load text from a .docx file into a single Sophons document."""

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
        metadata = {
            "source": str(self.path),
            "file_name": self.path.name,
            "mime_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        }
        metadata.update(self.metadata)

        return [
            Document(
                id=self.id,
                content=self._read_docx_text(),
                metadata=metadata,
            )
        ]

    def lazy_load(self):
        yield from self.load()

    def _read_docx_text(self) -> str:
        with ZipFile(self.path) as archive:
            xml_bytes = archive.read("word/document.xml")

        root = ElementTree.fromstring(xml_bytes)
        namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
        paragraphs: list[str] = []

        for paragraph in root.findall(".//w:p", namespace):
            text_parts = [
                node.text or ""
                for node in paragraph.findall(".//w:t", namespace)
            ]
            text = "".join(text_parts).strip()
            if text:
                paragraphs.append(text)

        return "\n".join(paragraphs)

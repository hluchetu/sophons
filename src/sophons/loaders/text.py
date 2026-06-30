from __future__ import annotations

from pathlib import Path
from typing import Any

from sophons.documents import Document


class TextLoader:
    """Load plain text into a single Sophons document."""

    def __init__(
        self,
        text: str,
        *,
        id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.text = text
        self.id = id
        self.metadata = dict(metadata or {})

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        encoding: str = "utf-8",
        id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> TextLoader:
        file_path = Path(path)
        file_metadata = {
            "source": str(file_path),
            "file_name": file_path.name,
        }
        file_metadata.update(metadata or {})

        return cls(
            file_path.read_text(encoding=encoding),
            id=id or str(file_path),
            metadata=file_metadata,
        )

    def load(self) -> list[Document]:
        return [
            Document(
                id=self.id,
                content=self.text,
                metadata=dict(self.metadata),
            )
        ]

    def lazy_load(self):
        yield from self.load()

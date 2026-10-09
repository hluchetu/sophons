from __future__ import annotations

import mimetypes
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any

# Suffixes the standard table misses or maps inconsistently across platforms.
_MIME_TYPES = {
    ".md": "text/markdown",
    ".markdown": "text/markdown",
    ".txt": "text/plain",
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".html": "text/html",
    ".htm": "text/html",
}


def guess_mime_type(name: str) -> str | None:
    """Mime type for a file name, or None when the suffix is not recognized."""

    suffix = Path(name).suffix.lower()
    return _MIME_TYPES.get(suffix) or mimetypes.guess_type(name)[0]


def normalize_mime_type(mime_type: str) -> str:
    """Lowercase a mime type and drop parameters such as ``; charset=utf-8``."""

    return mime_type.split(";", 1)[0].strip().lower()


@dataclass(frozen=True, slots=True)
class Blob:
    """Raw bytes and what they claim to be. Parsers never see paths or URLs."""

    data: bytes
    mime_type: str
    source: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.data, (bytes, bytearray)):
            raise TypeError("Blob data must be bytes")
        object.__setattr__(self, "data", bytes(self.data))
        object.__setattr__(self, "mime_type", normalize_mime_type(self.mime_type))
        if not self.mime_type:
            raise ValueError("Blob requires a mime type")

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        *,
        mime_type: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Blob:
        path = Path(path)
        resolved = mime_type or guess_mime_type(path.name)
        if resolved is None:
            raise ValueError(f"Cannot determine a mime type for {path.name!r}; pass mime_type")
        return cls(
            data=path.read_bytes(),
            mime_type=resolved,
            source=str(path),
            metadata=dict(metadata or {}),
        )

    @property
    def checksum(self) -> str:
        return sha256(self.data).hexdigest()

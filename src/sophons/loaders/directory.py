from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from sophons.documents import Document
from sophons.loaders.file import FileLoader, UnsupportedFileTypeError


class DirectoryLoader:
    """Load supported files from a directory."""

    def __init__(
        self,
        path: str | Path,
        *,
        glob: str = "**/*",
        metadata: dict[str, Any] | None = None,
        ignore_unsupported: bool = True,
    ) -> None:
        self.path = Path(path)
        self.glob = glob
        self.metadata = dict(metadata or {})
        self.ignore_unsupported = ignore_unsupported

    def load(self) -> list[Document]:
        return list(self.lazy_load())

    def lazy_load(self) -> Iterable[Document]:
        for file_path in sorted(self.path.glob(self.glob)):
            if not file_path.is_file():
                continue

            try:
                loader = FileLoader(
                    file_path,
                    metadata={
                        **self.metadata,
                        "directory": str(self.path),
                    },
                )
            except UnsupportedFileTypeError:
                if self.ignore_unsupported:
                    continue
                raise

            yield from loader.lazy_load()

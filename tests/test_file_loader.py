from __future__ import annotations

import pytest

from sophons.documents import Document
from sophons.errors import ErrorCode, UnsupportedFileTypeError
from sophons.loaders import FileLoader


def test_file_loader_routes_text_files(tmp_path) -> None:
    path = tmp_path / "cover_letter.txt"
    path.write_text("Dear hiring team", encoding="utf-8")

    assert FileLoader(path, metadata={"kind": "cover_letter"}).load() == [
        Document(
            id=str(path),
            content="Dear hiring team",
            metadata={
                "source": str(path),
                "file_name": "cover_letter.txt",
                "mime_type": "text/plain",
                "kind": "cover_letter",
            },
        )
    ]


def test_file_loader_rejects_unsupported_files(tmp_path) -> None:
    path = tmp_path / "image.png"
    path.write_bytes(b"not actually an image")

    with pytest.raises(UnsupportedFileTypeError, match="Unsupported file type") as exc_info:
        FileLoader(path).load()

    assert exc_info.value.error_code == ErrorCode.UNSUPPORTED_FILE_TYPE
    assert exc_info.value.details == {"path": str(path), "suffix": ".png"}

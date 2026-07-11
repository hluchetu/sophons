from __future__ import annotations

from sophons.documents import Document
from sophons.loaders import DirectoryLoader


def test_directory_loader_loads_supported_files_and_skips_unsupported(tmp_path) -> None:
    cv = tmp_path / "cv.md"
    cv.write_text("Python RAG experience", encoding="utf-8")
    ignored = tmp_path / "headshot.png"
    ignored.write_bytes(b"not loaded")

    loader = DirectoryLoader(tmp_path, metadata={"user_id": "user_1"})

    assert loader.load() == [
        Document(
            id=str(cv),
            content="Python RAG experience",
            metadata={
                "source": str(cv),
                "file_name": "cv.md",
                "mime_type": "text/markdown",
                "user_id": "user_1",
                "directory": str(tmp_path),
            },
        )
    ]

from __future__ import annotations

from sophons.documents import Document
from sophons.loaders import TextLoader


def test_text_loader_loads_text_as_document() -> None:
    loader = TextLoader(
        "hello world",
        id="doc_1",
        metadata={"source": "inline"},
    )

    assert loader.load() == [
        Document(
            id="doc_1",
            content="hello world",
            metadata={"source": "inline"},
        )
    ]


def test_text_loader_lazy_loads_text_as_document() -> None:
    loader = TextLoader("hello world")

    assert list(loader.lazy_load()) == [Document(content="hello world")]


def test_text_loader_loads_file(tmp_path) -> None:
    path = tmp_path / "cv.md"
    path.write_text("Python and RAG experience", encoding="utf-8")

    loader = TextLoader.from_file(path, metadata={"kind": "cv"})

    assert loader.load() == [
        Document(
            id=str(path),
            content="Python and RAG experience",
            metadata={
                "source": str(path),
                "file_name": "cv.md",
                "kind": "cv",
            },
        )
    ]

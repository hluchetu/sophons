from __future__ import annotations

import pytest

from sophons.documents import Document
from sophons.splitters import MarkdownSplitter


def test_markdown_splitter_splits_by_headings() -> None:
    splitter = MarkdownSplitter()
    document = Document(
        id="cv",
        content=(
            "# CV\n"
            "Profile summary.\n\n"
            "## Experience\n"
            "Built RAG systems.\n\n"
            "### Sophons\n"
            "Built loaders and splitters.\n\n"
            "## Skills\n"
            "Python, Playwright, RAG."
        ),
        metadata={"source": "cv.md"},
    )

    chunks = splitter.split_document(document)

    assert [chunk.id for chunk in chunks] == [
        "cv#section_0",
        "cv#section_1",
        "cv#section_2",
        "cv#section_3",
    ]
    assert [chunk.metadata["heading_path"] for chunk in chunks] == [
        ["CV"],
        ["CV", "Experience"],
        ["CV", "Experience", "Sophons"],
        ["CV", "Skills"],
    ]
    assert chunks[1].content == "## Experience\nBuilt RAG systems."


def test_markdown_splitter_keeps_preamble_without_heading_path() -> None:
    splitter = MarkdownSplitter()
    chunks = splitter.split_document(
        Document(id="cover", content="Dear team.\n\n# Why me\nPython experience.")
    )

    assert [chunk.content for chunk in chunks] == [
        "Dear team.",
        "# Why me\nPython experience.",
    ]
    assert [chunk.metadata["heading_path"] for chunk in chunks] == [[], ["Why me"]]


def test_markdown_splitter_rejects_invalid_heading_level() -> None:
    with pytest.raises(ValueError, match="max_heading_level"):
        MarkdownSplitter(max_heading_level=0)

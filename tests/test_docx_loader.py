from __future__ import annotations

from zipfile import ZipFile

from sophons.documents import Document
from sophons.loaders import DocxLoader


def test_docx_loader_loads_paragraph_text(tmp_path) -> None:
    path = tmp_path / "cv.docx"
    document_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    <w:p><w:r><w:t>Python experience</w:t></w:r></w:p>
    <w:p><w:r><w:t>RAG systems</w:t></w:r></w:p>
  </w:body>
</w:document>
"""
    with ZipFile(path, "w") as archive:
        archive.writestr("word/document.xml", document_xml)

    loader = DocxLoader(path, metadata={"kind": "cv"})

    assert loader.load() == [
        Document(
            id=str(path),
            content="Python experience\nRAG systems",
            metadata={
                "source": str(path),
                "file_name": "cv.docx",
                "mime_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                "kind": "cv",
            },
        )
    ]

from __future__ import annotations

import pytest

from sophons.errors import ErrorCode, MissingDependencyError
from sophons.loaders import PDFLoader


def test_pdf_loader_reports_missing_optional_dependency(tmp_path) -> None:
    path = tmp_path / "cv.pdf"
    path.write_bytes(b"%PDF-1.4\n")

    loader = PDFLoader(path)

    with pytest.raises(MissingDependencyError, match="PDFLoader requires pypdf") as exc_info:
        loader.load()

    assert exc_info.value.error_code == ErrorCode.MISSING_DEPENDENCY
    assert exc_info.value.details == {"dependency": "pypdf", "extra": "pdf"}

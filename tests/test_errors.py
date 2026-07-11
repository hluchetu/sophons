from __future__ import annotations

from sophons.errors import ConfigurationError, ErrorCode, SophonsError


def test_sophons_error_carries_code_and_details() -> None:
    error = SophonsError(
        "Something failed.",
        error_code=ErrorCode.RETRIEVER_ERROR,
        details={"retriever": "bm25"},
    )

    assert str(error) == "Something failed."
    assert error.error_code == ErrorCode.RETRIEVER_ERROR
    assert error.details == {"retriever": "bm25"}


def test_configuration_error_uses_configuration_error_code() -> None:
    error = ConfigurationError(
        "Invalid setting.",
        details={"parameter": "chunk_size"},
    )

    assert error.error_code == ErrorCode.CONFIGURATION_ERROR
    assert error.details == {"parameter": "chunk_size"}

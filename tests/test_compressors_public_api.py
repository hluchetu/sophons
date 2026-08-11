from __future__ import annotations

import subprocess
import sys

import pytest


def test_public_api_is_importable_without_sentence_transformers() -> None:
    code = """
import sys
import sophons.rag.compressors as compressors

assert "sentence_transformers" not in sys.modules
assert "DocumentCompressor" in compressors.__all__
assert "CrossEncoderReranker" not in compressors.__all__
assert "CrossEncoderReranker" in dir(compressors)

from sophons.rag.compressors import CrossEncoderReranker

assert CrossEncoderReranker.__name__ == "CrossEncoderReranker"
assert "sentence_transformers" not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_unknown_public_name_raises_attribute_error() -> None:
    import sophons.rag.compressors as compressors

    with pytest.raises(AttributeError, match="MissingCompressor"):
        getattr(compressors, "MissingCompressor")


def test_legacy_integration_import_remains_available() -> None:
    from sophons.integrations.compressors import (
        CrossEncoderReranker as IntegrationReranker,
    )
    from sophons.rag.compressors import CrossEncoderReranker as PublicReranker

    assert IntegrationReranker is PublicReranker

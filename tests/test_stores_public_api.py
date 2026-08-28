from __future__ import annotations

import subprocess
import sys

import pytest

from sophons import Document
from sophons.stores import VectorStore


def test_public_store_api_is_importable_without_chromadb() -> None:
    code = """
import sys
import sophons.stores as stores

assert "chromadb" not in sys.modules
assert "InMemoryVectorStore" in stores.__all__
assert "NumPyVectorStore" in stores.__all__
assert "ChromaVectorStore" in dir(stores)

from sophons.stores import ChromaVectorStore, InMemoryVectorStore, NumPyVectorStore

assert ChromaVectorStore.__name__ == "ChromaVectorStore"
assert InMemoryVectorStore.__name__ == "InMemoryVectorStore"
assert NumPyVectorStore.__name__ == "NumPyVectorStore"
assert "chromadb" not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_unknown_public_store_name_raises_attribute_error() -> None:
    import sophons.stores as stores

    with pytest.raises(AttributeError, match="MissingStore"):
        getattr(stores, "MissingStore")


def test_in_memory_store_matches_public_protocol() -> None:
    from sophons.stores import InMemoryVectorStore

    store = InMemoryVectorStore()
    assert isinstance(store, VectorStore)


def test_in_memory_store_add_search_and_delete() -> None:
    from sophons.stores import InMemoryVectorStore

    first = Document(id="first", content="first")
    second = Document(id="second", content="second")
    store = InMemoryVectorStore()

    store.add([first, second], [[1.0, 0.0], [0.0, 1.0]])
    assert [document.id for document in store.search([1.0, 0.0])] == [
        "first",
        "second",
    ]

    store.delete(["first"])
    assert [document.id for document in store.search([1.0, 0.0])] == ["second"]


def test_legacy_integration_store_import_remains_available() -> None:
    from sophons.integrations.vector_stores import (
        InMemoryVectorStore as IntegrationStore,
    )
    from sophons.stores import InMemoryVectorStore as PublicStore

    assert IntegrationStore is PublicStore

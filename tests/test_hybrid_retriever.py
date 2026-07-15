from __future__ import annotations

import pytest

from sophons.documents import Document
from sophons.retrieval import HybridRetriever, Retriever


def _doc(id_: str) -> Document:
    return Document(id=id_, content=f"content of {id_}")


class Scripted:
    """A retriever that returns a fixed ranked list, ignoring the query."""

    def __init__(self, ids: list[str]) -> None:
        self._ids = ids

    def retrieve(self, query: str, *, limit: int = 10) -> list[Document]:
        return [_doc(i) for i in self._ids[:limit]]


def test_consensus_beats_single_list_first_place() -> None:
    # "a" is first on one list only; "b" is second on BOTH.
    left = Scripted(["a", "b", "c"])
    right = Scripted(["d", "b", "e"])

    results = HybridRetriever([left, right]).retrieve("q", limit=3)

    assert results[0].id == "b"  # ranked well by both -> wins


def test_documents_deduplicate_across_lists() -> None:
    results = HybridRetriever(
        [Scripted(["a", "b"]), Scripted(["b", "a"])]
    ).retrieve("q", limit=10)

    assert [d.id for d in results].count("a") == 1
    assert [d.id for d in results].count("b") == 1


def test_limit_is_respected_after_fusion() -> None:
    results = HybridRetriever(
        [Scripted(["a", "b", "c", "d", "e"])]
    ).retrieve("q", limit=2)

    assert len(results) == 2
    assert [d.id for d in results] == ["a", "b"]


def test_single_retriever_preserves_its_order() -> None:
    results = HybridRetriever([Scripted(["x", "y", "z"])]).retrieve("q", limit=3)

    assert [d.id for d in results] == ["x", "y", "z"]


def test_empty_retriever_list_raises() -> None:
    with pytest.raises(ValueError, match="at least one"):
        HybridRetriever([])


def test_satisfies_retriever_protocol() -> None:
    assert isinstance(HybridRetriever([Scripted(["a"])]), Retriever)

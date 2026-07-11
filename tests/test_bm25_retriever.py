from __future__ import annotations

import pytest

from sophons.documents import Document
from sophons.errors import ConfigurationError, ErrorCode
from sophons.retrieval import BM25Retriever, Retriever, default_tokenizer


def test_default_tokenizer_lowercases_and_extracts_words() -> None:
    assert default_tokenizer("Python, RAG & Browser-Automation!") == [
        "python",
        "rag",
        "browser",
        "automation",
    ]


def test_bm25_retriever_returns_matching_documents_with_scores() -> None:
    retriever = BM25Retriever.from_documents(
        [
            Document(id="cv#1", content="Python RAG browser automation"),
            Document(id="cv#2", content="Frontend design systems"),
            Document(id="job#1", content="Greenhouse Python automation role"),
        ]
    )

    results = retriever.retrieve("python automation", limit=2)

    assert isinstance(retriever, Retriever)
    assert [result.id for result in results] == ["cv#1", "job#1"]
    assert all(result.score is not None and result.score > 0 for result in results)


def test_bm25_retriever_returns_empty_for_no_matches() -> None:
    retriever = BM25Retriever([Document(id="cv#1", content="Python RAG")])

    assert retriever.retrieve("kubernetes") == []


def test_bm25_retriever_respects_limit() -> None:
    retriever = BM25Retriever(
        [
            Document(id="doc_1", content="python"),
            Document(id="doc_2", content="python"),
        ]
    )

    assert len(retriever.retrieve("python", limit=1)) == 1
    assert retriever.retrieve("python", limit=0) == []


def test_bm25_retriever_rejects_invalid_parameters() -> None:
    with pytest.raises(ConfigurationError, match="k1") as k1_error:
        BM25Retriever([], k1=0)

    assert k1_error.value.error_code == ErrorCode.CONFIGURATION_ERROR
    assert k1_error.value.details == {"parameter": "k1", "value": 0}

    with pytest.raises(ConfigurationError, match="b") as b_error:
        BM25Retriever([], b=2)

    assert b_error.value.error_code == ErrorCode.CONFIGURATION_ERROR
    assert b_error.value.details == {"parameter": "b", "value": 2}

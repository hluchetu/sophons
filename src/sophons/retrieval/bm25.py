from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Callable, Iterable

from opentelemetry import trace

from sophons.documents import Document
from sophons.errors import ConfigurationError
from sophons.observability import _semconv

Tokenizer = Callable[[str], list[str]]

_TRACER = trace.get_tracer("sophons.retrieval")


class BM25Retriever:
    """Lexical BM25 retriever for small/local document collections."""

    def __init__(
        self,
        documents: Iterable[Document],
        *,
        tokenizer: Tokenizer | None = None,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if k1 <= 0:
            raise ConfigurationError(
                "k1 must be greater than 0.",
                details={"parameter": "k1", "value": k1},
            )
        if b < 0 or b > 1:
            raise ConfigurationError(
                "b must be between 0 and 1.",
                details={"parameter": "b", "value": b},
            )

        self.documents = list(documents)
        self.tokenizer = tokenizer or default_tokenizer
        self.k1 = k1
        self.b = b
        self._tokenized_documents = [
            self.tokenizer(document.content) for document in self.documents
        ]
        self._term_frequencies = [
            Counter(tokens) for tokens in self._tokenized_documents
        ]
        self._document_lengths = [
            len(tokens) for tokens in self._tokenized_documents
        ]
        self._average_document_length = (
            sum(self._document_lengths) / len(self._document_lengths)
            if self._document_lengths
            else 0.0
        )
        self._document_frequencies = self._build_document_frequencies()

    @classmethod
    def from_documents(
        cls,
        documents: Iterable[Document],
        *,
        tokenizer: Tokenizer | None = None,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> BM25Retriever:
        return cls(documents, tokenizer=tokenizer, k1=k1, b=b)

    def retrieve(self, query: str, *, limit: int = 10) -> list[Document]:
        with _TRACER.start_as_current_span(
            "retriever.search",
            attributes={_semconv.RETRIEVER: "bm25", _semconv.LIMIT: limit},
        ) as span:
            results = self._retrieve(query, limit=limit)
            span.set_attribute(_semconv.RESULT_COUNT, len(results))
            return results

    def _retrieve(self, query: str, *, limit: int) -> list[Document]:
        if limit <= 0:
            return []

        query_tokens = self.tokenizer(query)
        if not query_tokens or not self.documents:
            return []

        scored = [
            (score, document)
            for document, score in zip(
                self.documents,
                self._score_documents(query_tokens),
                strict=True,
            )
            if score > 0
        ]
        scored.sort(key=lambda item: item[0], reverse=True)

        return [
            document.with_score(score)
            for score, document in scored[:limit]
        ]

    def _score_documents(self, query_tokens: list[str]) -> list[float]:
        return [
            self._score_document(query_tokens, index)
            for index in range(len(self.documents))
        ]

    def _score_document(self, query_tokens: list[str], index: int) -> float:
        score = 0.0
        frequencies = self._term_frequencies[index]
        document_length = self._document_lengths[index]

        for token in query_tokens:
            term_frequency = frequencies[token]
            if term_frequency == 0:
                continue

            idf = self._idf(token)
            denominator = term_frequency + self.k1 * (
                1
                - self.b
                + self.b * document_length / self._average_document_length
            )
            score += idf * (
                term_frequency * (self.k1 + 1)
            ) / denominator

        return score

    def _idf(self, token: str) -> float:
        document_count = len(self.documents)
        document_frequency = self._document_frequencies.get(token, 0)
        return math.log(
            1 + (document_count - document_frequency + 0.5) / (document_frequency + 0.5)
        )

    def _build_document_frequencies(self) -> dict[str, int]:
        frequencies: dict[str, int] = {}
        for tokens in self._tokenized_documents:
            for token in set(tokens):
                frequencies[token] = frequencies.get(token, 0) + 1
        return frequencies


def default_tokenizer(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())

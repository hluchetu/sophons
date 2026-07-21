from __future__ import annotations

from opentelemetry import trace

from sophons.documents import Document
from sophons.errors import ConfigurationError
from sophons.models import ChatModel, Message
from sophons.observability import _semconv
from sophons.retrieval.base import Retriever

_TRACER = trace.get_tracer("sophons.retrieval")

DEFAULT_REWRITE_PROMPT = """\
Generate {count} different versions of the given question to retrieve \
relevant documents from a search index. Provide one query per line, with \
no numbering.

Question: {question}"""


class MultiQueryRetriever:
    """
    Generates several query rewrites, retrieves documents for each rewrite,
    and returns a deduplicated list of Documents.

    This satisfies the Retriever protocol:

        retrieve(query, *, limit) -> list[Document]

    Use ``retrieve_with_queries()`` when you also want to inspect the generated
    queries.
    """

    def __init__(
        self,
        retriever: Retriever,
        model: ChatModel,
        *,
        prompt: str = DEFAULT_REWRITE_PROMPT,
        rewrite_count: int = 3,
        include_original: bool = True,
    ) -> None:
        if rewrite_count <= 0:
            raise ConfigurationError(
                "rewrite_count must be greater than 0.",
                details={"parameter": "rewrite_count", "value": rewrite_count},
            )

        self._retriever = retriever
        self._model = model
        self._prompt = prompt
        self._rewrite_count = rewrite_count
        self._include_original = include_original
        self.last_queries: list[str] = []

    def generate_queries(self, question: str) -> list[str]:
        """Generate query rewrites without retrieving documents."""
        response = self._model.invoke(
            [
                Message(
                    role="user",
                    content=self._prompt.format(
                        count=self._rewrite_count,
                        question=question,
                    ),
                )
            ]
        )

        rewrites = [
            line.removeprefix("-").strip()
            for line in response.content.splitlines()
            if line.strip()
        ][: self._rewrite_count]

        if self._include_original:
            rewrites.append(question)

        return self._dedupe_queries(rewrites)

    def retrieve(self, query: str, *, limit: int = 10) -> list[Document]:
        with _TRACER.start_as_current_span(
            "retriever.search",
            attributes={
                _semconv.RETRIEVER: "multi_query",
                _semconv.LIMIT: limit,
            },
        ) as span:
            queries = self.generate_queries(query)
            self.last_queries = queries

            span.set_attribute(_semconv.QUERY_COUNT, len(queries))
            span.set_attribute(_semconv.INCLUDE_ORIGINAL, self._include_original)

            results = self._retrieve_for_queries(queries, limit=limit)

            span.set_attribute(_semconv.RESULT_COUNT, len(results))
            return results

    def retrieve_with_queries(
        self,
        query: str,
        *,
        limit: int = 10,
    ) -> tuple[list[str], list[Document]]:
        """Retrieve documents and also return the generated queries."""
        documents = self.retrieve(query, limit=limit)
        return self.last_queries, documents

    def _retrieve_for_queries(
        self,
        queries: list[str],
        *,
        limit: int,
    ) -> list[Document]:
        if limit <= 0:
            return []

        seen: set[str] = set()
        results: list[Document] = []

        for rewritten_query in queries:
            documents = self._retriever.retrieve(rewritten_query, limit=limit)

            for document in documents:
                key = document.id or document.content
                if key in seen:
                    continue

                seen.add(key)
                results.append(document.with_metadata(matched_query=rewritten_query))

        return results[:limit]

    def _dedupe_queries(self, queries: list[str]) -> list[str]:
        seen: set[str] = set()
        deduped: list[str] = []

        for query in queries:
            if query in seen:
                continue
            seen.add(query)
            deduped.append(query)

        return deduped

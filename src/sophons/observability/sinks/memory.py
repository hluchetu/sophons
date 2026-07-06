from __future__ import annotations

from sophons.observability.base import Span


class InMemorySink:
    """Stores spans in a list for inspection after a run. For tests and evals."""

    def __init__(self) -> None:
        self.spans: list[Span] = []

    def emit(self, span: Span) -> None:
        self.spans.append(span)

    def clear(self) -> None:
        self.spans.clear()

from __future__ import annotations

import sys
from collections import defaultdict
from typing import TextIO

from sophons.observability.base import Span

# Compact labels for common attributes, matching the trace tree output.
_ATTR_LABELS = {
    "input_tokens": "in",
    "output_tokens": "out",
    "result_count": "results",
}
_SKIP_ATTRS = {"session_id", "tool_name"}


class ConsoleSink:
    """
    Pretty-prints span trees to stdout. For development.

    Spans close child-first, so spans are buffered per trace and the
    full tree is printed once its root span closes.

    Output:
        [agent.run]           ✓  2341ms
          [model.call]        ✓   312ms  in=312 out=24
          [tool.search_docs]  ✓   148ms
            [retriever.search] ✓  136ms  results=4
    """

    def __init__(self, stream: TextIO | None = None) -> None:
        self._stream = stream if stream is not None else sys.stdout
        self._buffer: dict[str, list[Span]] = defaultdict(list)

    def emit(self, span: Span) -> None:
        self._buffer[span.trace_id].append(span)
        if span.parent_span_id is None:
            self._print_tree(span.trace_id)

    def _print_tree(self, trace_id: str) -> None:
        spans = self._buffer.pop(trace_id)
        children: dict[str | None, list[Span]] = defaultdict(list)
        for span in spans:
            children[span.parent_span_id].append(span)

        def walk(span: Span, depth: int) -> None:
            self._stream.write(self._format(span, depth) + "\n")
            for child in sorted(
                children.get(span.span_id, []), key=lambda s: s.start_time
            ):
                walk(child, depth + 1)

        for root in sorted(children[None], key=lambda s: s.start_time):
            walk(root, 0)

    def _format(self, span: Span, depth: int) -> str:
        mark = "✓" if span.status == "ok" else "✗"
        label = f"{'  ' * depth}[{span.name}]"
        line = f"{label:<28} {mark} {span.duration_ms:6.0f}ms"
        attrs = " ".join(
            f"{_ATTR_LABELS.get(key, key)}={value}"
            for key, value in span.attributes.items()
            if key not in _SKIP_ATTRS and isinstance(value, (int, float, bool))
        )
        if attrs:
            line += f"  {attrs}"
        if span.error:
            line += f"  error={span.error}"
        return line

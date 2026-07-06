from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

# ── Span kinds ─────────────────────────────────────────────────────────────────


class SpanKind:
    AGENT = "agent"
    MODEL = "model"
    TOOL = "tool"
    RETRIEVER = "retriever"
    LOADER = "loader"
    SPLITTER = "splitter"
    MEMORY = "memory"
    STORE = "store"
    INTERNAL = "internal"


# ── Span ───────────────────────────────────────────────────────────────────────


@dataclass
class Span:
    """One unit of timed work. Maps 1:1 to an OpenTelemetry span."""

    span_id: str
    trace_id: str
    parent_span_id: str | None
    name: str
    kind: str
    start_time: float
    end_time: float | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    status: Literal["ok", "error"] = "ok"
    error: str | None = None

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value

    @property
    def duration_ms(self) -> float:
        if self.end_time is None:
            return 0.0
        return (self.end_time - self.start_time) * 1000


# ── Sink ───────────────────────────────────────────────────────────────────────


class Sink(Protocol):
    """Receives completed spans. Implement this to add a new backend."""

    def emit(self, span: Span) -> None: ...

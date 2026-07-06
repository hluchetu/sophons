from __future__ import annotations

import json
import sys
import urllib.request
from collections import defaultdict
from typing import Any

from sophons.observability.base import Span

# OTEL semantic conventions — sophons attribute name → OTEL name.
_SEMCONV = {
    "model_name": "gen_ai.request.model",
    "input_tokens": "gen_ai.usage.input_tokens",
    "output_tokens": "gen_ai.usage.output_tokens",
    "tool_name": "gen_ai.tool.name",
}

_STATUS_OK = 1
_STATUS_ERROR = 2


class OTELSink:
    """
    Exports spans to any OTLP-compatible backend (Jaeger, Grafana Tempo,
    Datadog, Langfuse OTEL endpoint, ...) over OTLP/HTTP JSON.

    Spans are buffered per trace and exported in one request when the
    trace's root span closes. Export failures are reported to stderr but
    never raised — observability must not break the agent.

    Usage:
        sink = OTELSink(endpoint="http://localhost:4318")
        tracer = Tracer(sinks=[sink])
    """

    def __init__(
        self,
        endpoint: str = "http://localhost:4318",
        service_name: str = "sophons",
        timeout: float = 5.0,
    ) -> None:
        self._url = endpoint.rstrip("/") + "/v1/traces"
        self._service_name = service_name
        self._timeout = timeout
        self._buffer: dict[str, list[Span]] = defaultdict(list)

    def emit(self, span: Span) -> None:
        self._buffer[span.trace_id].append(span)
        if span.parent_span_id is None:
            self._export(self._buffer.pop(span.trace_id))

    def _export(self, spans: list[Span]) -> None:
        payload = {
            "resourceSpans": [
                {
                    "resource": {
                        "attributes": [
                            {
                                "key": "service.name",
                                "value": {"stringValue": self._service_name},
                            }
                        ]
                    },
                    "scopeSpans": [
                        {
                            "scope": {"name": "sophons.observability"},
                            "spans": [_to_otlp(s) for s in spans],
                        }
                    ],
                }
            ]
        }
        request = urllib.request.Request(
            self._url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout):
                pass
        except Exception as exc:
            print(f"OTELSink: export failed: {exc}", file=sys.stderr)


def _to_otlp(span: Span) -> dict[str, Any]:
    attributes = [
        {"key": "sophons.kind", "value": {"stringValue": span.kind}},
        *(
            {"key": _SEMCONV.get(key, key), "value": _to_otlp_value(value)}
            for key, value in span.attributes.items()
        ),
    ]
    status: dict[str, Any] = {"code": _STATUS_OK}
    if span.status == "error":
        status = {"code": _STATUS_ERROR, "message": span.error or ""}
    return {
        "traceId": span.trace_id,
        "spanId": span.span_id,
        "parentSpanId": span.parent_span_id or "",
        "name": span.name,
        "kind": 1,  # SPAN_KIND_INTERNAL
        "startTimeUnixNano": str(int(span.start_time * 1_000_000_000)),
        "endTimeUnixNano": str(int((span.end_time or span.start_time) * 1_000_000_000)),
        "attributes": attributes,
        "status": status,
    }


def _to_otlp_value(value: Any) -> dict[str, Any]:
    if isinstance(value, bool):
        return {"boolValue": value}
    if isinstance(value, int):
        return {"intValue": str(value)}
    if isinstance(value, float):
        return {"doubleValue": value}
    return {"stringValue": str(value)}

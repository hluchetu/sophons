from __future__ import annotations

import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any

from sophons.observability.base import Sink, Span, SpanKind

if TYPE_CHECKING:
    from sophons.agents.hooks import (
        AfterModelCall,
        AfterToolCall,
        AgentFailed,
        AgentFinished,
        AgentStarted,
        BeforeModelCall,
        BeforeToolCall,
        HookRegistry,
    )

# ── Active span context ────────────────────────────────────────────────────────

_active_span: ContextVar[Span | None] = ContextVar("_active_span", default=None)

_REDACTED_SUFFIXES = ("_content", "_text", "_prompt", "_response")


# ── Tracer ─────────────────────────────────────────────────────────────────────


class Tracer:
    """
    Creates spans, tracks the active span via a context variable,
    and emits completed spans to every sink.

    Usage:
        tracer = Tracer(sinks=[ConsoleSink()])

        with tracer.span("retriever.search", kind="retriever", limit=5) as span:
            results = retriever.retrieve(query)
            span.set_attribute("result_count", len(results))
    """

    def __init__(self, sinks: list[Sink], *, redact: bool = True) -> None:
        self._sinks = sinks
        self._redact = redact

    @contextmanager
    def span(
        self,
        name: str,
        kind: str = SpanKind.INTERNAL,
        **attributes: Any,
    ) -> Iterator[Span]:
        parent = _active_span.get()
        span = self._start(name, kind, parent, **attributes)
        token = _active_span.set(span)
        try:
            yield span
        except Exception as exc:
            span.status = "error"
            span.error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            span.end_time = time.time()
            _active_span.reset(token)
            self._emit(span)

    def hooks(self) -> HookRegistry:
        """
        Return a HookRegistry pre-wired to translate agent hook events
        into spans on this tracer.

        Usage:
            agent = Agent(model=model, tools=tools, hooks=tracer.hooks())
        """
        from sophons.agents import hooks as agent_hooks

        bridge = _HookBridge(self)
        registry = agent_hooks.HookRegistry()
        registry.register(agent_hooks.AgentStarted, bridge.on_agent_started)
        registry.register(agent_hooks.AgentFinished, bridge.on_agent_finished)
        registry.register(agent_hooks.AgentFailed, bridge.on_agent_failed)
        registry.register(agent_hooks.BeforeModelCall, bridge.on_before_model_call)
        registry.register(agent_hooks.AfterModelCall, bridge.on_after_model_call)
        registry.register(agent_hooks.BeforeToolCall, bridge.on_before_tool_call)
        registry.register(agent_hooks.AfterToolCall, bridge.on_after_tool_call)
        return registry

    # ── Internal ───────────────────────────────────────────────────────────────

    def _start(
        self,
        name: str,
        kind: str,
        parent: Span | None,
        **attributes: Any,
    ) -> Span:
        return Span(
            span_id=uuid.uuid4().hex[:16],
            trace_id=parent.trace_id if parent else uuid.uuid4().hex,
            parent_span_id=parent.span_id if parent else None,
            name=name,
            kind=kind,
            start_time=time.time(),
            attributes=dict(attributes),
        )

    def _finish(self, span: Span, *, error: str | None = None) -> None:
        span.end_time = time.time()
        if error is not None:
            span.status = "error"
            span.error = error
        self._emit(span)

    def _emit(self, span: Span) -> None:
        if self._redact:
            for key, value in span.attributes.items():
                if key.endswith(_REDACTED_SUFFIXES) and value is not None:
                    span.attributes[key] = "[redacted]"
        for sink in self._sinks:
            sink.emit(span)


# ── maybe_span ─────────────────────────────────────────────────────────────────


class _NoopSpan:
    """Stands in for a Span when no tracer is configured."""

    def set_attribute(self, key: str, value: Any) -> None: ...


_NOOP_SPAN = _NoopSpan()


@contextmanager
def maybe_span(
    tracer: Tracer | None,
    name: str,
    kind: str = SpanKind.INTERNAL,
    **attributes: Any,
) -> Iterator[Span | _NoopSpan]:
    """
    Trace the block when a tracer is present; no-op otherwise.

    Lets components accept ``tracer: Tracer | None = None`` without
    branching at every call site:

        with maybe_span(self._tracer, "retriever.search", kind="retriever") as span:
            results = ...
            span.set_attribute("result_count", len(results))
    """
    if tracer is None:
        yield _NOOP_SPAN
    else:
        with tracer.span(name, kind, **attributes) as span:
            yield span


# ── Hook bridge ────────────────────────────────────────────────────────────────


class _HookBridge:
    """
    Translates agent hook events into spans.

    Hook events arrive as separate before/after callbacks rather than nested
    `with` blocks, so open spans are tracked by step (model calls) and
    tool_use_id (tool calls) until their closing event arrives.
    """

    def __init__(self, tracer: Tracer) -> None:
        self._tracer = tracer
        self._root: Span | None = None
        self._model_spans: dict[int, Span] = {}
        self._tool_spans: dict[str, Span] = {}

    def on_agent_started(self, event: AgentStarted) -> None:
        root = self._tracer._start("agent.run", SpanKind.AGENT, parent=None)
        if event.session_id is not None:
            root.set_attribute("session_id", event.session_id)
        self._root = root
        _active_span.set(root)

    def on_before_model_call(self, event: BeforeModelCall) -> None:
        if self._root is None:
            return
        self._model_spans[event.step] = self._tracer._start(
            "model.call", SpanKind.MODEL, parent=self._root, step=event.step
        )

    def on_after_model_call(self, event: AfterModelCall) -> None:
        span = self._model_spans.pop(event.step, None)
        if span is None:
            return
        if event.message is not None:
            usage = event.message.metadata.get("usage", {})
            if usage:
                span.set_attribute("input_tokens", usage.get("input_tokens", 0))
                span.set_attribute("output_tokens", usage.get("output_tokens", 0))
                if "cache_read_tokens" in usage:
                    span.set_attribute(
                        "cache_read_tokens", usage["cache_read_tokens"]
                    )
                if "cache_write_tokens" in usage:
                    span.set_attribute(
                        "cache_write_tokens", usage["cache_write_tokens"]
                    )
        self._tracer._finish(span)

    def on_before_tool_call(self, event: BeforeToolCall) -> None:
        if self._root is None:
            return
        span = self._tracer._start(
            f"tool.{event.tool_use.name}",
            SpanKind.TOOL,
            parent=self._root,
            tool_name=event.tool_use.name,
            step=event.step,
        )
        self._tool_spans[event.tool_use.tool_use_id] = span
        _active_span.set(span)

    def on_after_tool_call(self, event: AfterToolCall) -> None:
        span = self._tool_spans.pop(event.tool_use.tool_use_id, None)
        if span is None:
            return
        error = (
            event.tool_result.content
            if event.tool_result.status == "error"
            else None
        )
        self._tracer._finish(span, error=error)
        _active_span.set(self._root)

    def on_agent_finished(self, event: AgentFinished) -> None:
        root = self._root
        if root is None:
            return
        metrics = event.result.metrics
        root.attributes.update(
            steps=metrics.steps,
            model_calls=metrics.model_calls,
            tool_calls=metrics.tool_calls,
            input_tokens=metrics.input_tokens,
            output_tokens=metrics.output_tokens,
        )
        self._root = None
        _active_span.set(None)
        self._tracer._finish(root)

    def on_agent_failed(self, event: AgentFailed) -> None:
        root = self._root
        if root is None:
            return
        self._root = None
        _active_span.set(None)
        self._tracer._finish(
            root, error=f"{type(event.error).__name__}: {event.error}"
        )

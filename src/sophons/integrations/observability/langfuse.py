from __future__ import annotations

from typing import Any

from langfuse import Langfuse

from sophons.observability.base import Span, SpanKind

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


# Sophons span kind → Langfuse observation type.
_AS_TYPE = {
    SpanKind.AGENT: "agent",
    SpanKind.MODEL: "generation",
    SpanKind.TOOL: "tool",
}


class LangfuseSink:
    """
    Sends completed spans to Langfuse (v4 SDK) as a structured trace.

    kind=agent → trace root (agent observation)
    kind=model → generation (includes token usage)
    kind=tool  → tool observation
    all others → span observation

    Usage::

        from sophons.observability import Tracer
        from sophons.integrations.observability.langfuse import LangfuseSink

        tracer = Tracer(sinks=[LangfuseSink(
            public_key="pk-lf-...",
            secret_key="sk-lf-...",
            model_name="deepseek-chat",
        )])
        agent = Agent(model=model, tools=tools, hooks=tracer.hooks())
    """

    def __init__(
        self,
        public_key: str,
        secret_key: str,
        host: str = "https://cloud.langfuse.com",
        model_name: str = "unknown",
    ) -> None:
        self._lf = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=host,
        )
        self._model_name = model_name

    def emit(self, span: Span) -> None:
        as_type = _AS_TYPE.get(span.kind, "span")
        trace_context: dict[str, Any] = {"trace_id": span.trace_id}
        if span.parent_span_id is not None:
            trace_context["parent_span_id"] = span.parent_span_id

        observation = self._lf.start_observation(
            name=span.name,
            as_type=as_type,
            trace_context=trace_context,
        )

        update: dict[str, Any] = {
            "metadata": {
                **span.attributes,
                "kind": span.kind,
                "start_time": span.start_time,
                "duration_ms": round(span.duration_ms, 1),
            },
        }
        if as_type == "generation":
            update["model"] = span.attributes.get("model_name", self._model_name)
            if "input_tokens" in span.attributes or "output_tokens" in span.attributes:
                update["usage_details"] = {
                    "input": span.attributes.get("input_tokens", 0),
                    "output": span.attributes.get("output_tokens", 0),
                }
        if span.status == "error":
            update["level"] = "ERROR"
            update["status_message"] = span.error

        observation.update(**update)
        observation.end()

        # Root span closing means the trace is complete.
        if span.parent_span_id is None:
            self._lf.flush()


class LangfuseObservability:
    """
    Sends agent lifecycle events to Langfuse (v4 SDK).

    Each agent run becomes one Langfuse trace (root agent span).
    Model calls become generation observations with latency and token usage.
    Tool calls become tool observations nested inside the trace.

    Usage::

        obs = LangfuseObservability(
            public_key="pk-lf-...",
            secret_key="sk-lf-...",
            host="http://localhost:3003",
            model_name="deepseek-chat",
        )

        agent = Agent(model=model, tools=tools, hooks=obs.hooks())
    """

    def __init__(
        self,
        public_key: str,
        secret_key: str,
        host: str = "https://cloud.langfuse.com",
        model_name: str = "unknown",
    ) -> None:
        self._lf = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=host,
        )
        self._model_name = model_name
        self._trace_id: str | None = None
        self._root_span = None
        self._current_generation = None
        self._tool_spans: dict = {}

    def hooks(self) -> HookRegistry:
        registry = HookRegistry()
        registry.register(AgentStarted, self._on_started)
        registry.register(BeforeModelCall, self._on_before_model)
        registry.register(AfterModelCall, self._on_after_model)
        registry.register(BeforeToolCall, self._on_before_tool)
        registry.register(AfterToolCall, self._on_after_tool)
        registry.register(AgentFinished, self._on_finished)
        registry.register(AgentFailed, self._on_failed)
        return registry

    def _on_started(self, event: AgentStarted) -> None:
        self._trace_id = self._lf.create_trace_id()
        self._root_span = self._lf.start_observation(
            name="react-agent",
            as_type="agent",
            trace_context={"trace_id": self._trace_id},
            input=event.input,
            metadata={"session_id": event.session_id} if event.session_id else None,
        )

    def _on_before_model(self, event: BeforeModelCall) -> None:
        if not self._trace_id:
            return
        self._current_generation = self._lf.start_observation(
            name=f"llm-step-{event.step}",
            as_type="generation",
            trace_context={"trace_id": self._trace_id},
            input=[{"role": m.role, "content": m.content} for m in event.messages],
            model=self._model_name,
        )

    def _on_after_model(self, event: AfterModelCall) -> None:
        if not self._current_generation:
            return
        usage = event.message.metadata.get("usage", {})
        self._current_generation.update(
            output=event.message.content,
            usage_details={
                "input": usage.get("input_tokens", 0),
                "output": usage.get("output_tokens", 0),
            } if usage else None,
            metadata={"duration_ms": round(event.duration_ms, 1)},
        )
        self._current_generation.end()
        self._current_generation = None

    def _on_before_tool(self, event: BeforeToolCall) -> None:
        if not self._trace_id:
            return
        span = self._lf.start_observation(
            name=event.tool_use.name,
            as_type="tool",
            trace_context={"trace_id": self._trace_id},
            input=event.tool_use.input,
        )
        self._tool_spans[event.tool_use.tool_use_id] = span

    def _on_after_tool(self, event: AfterToolCall) -> None:
        span = self._tool_spans.pop(event.tool_use.tool_use_id, None)
        if span:
            span.update(
                output=event.tool_result.content,
                metadata={
                    "status": event.tool_result.status,
                    "duration_ms": round(event.duration_ms, 1),
                },
            )
            span.end()

    def _on_finished(self, event: AgentFinished) -> None:
        if not self._root_span:
            return
        self._root_span.update(
            output=event.result.message,
            metadata={
                "steps": event.result.metrics.steps,
                "model_calls": event.result.metrics.model_calls,
                "tool_calls": event.result.metrics.tool_calls,
                "duration_ms": round(event.result.metrics.duration_ms, 1),
            },
        )
        self._root_span.end()
        self._lf.flush()
        self._trace_id = None
        self._root_span = None

    def _on_failed(self, event: AgentFailed) -> None:
        if not self._root_span:
            return
        self._root_span.update(
            level="ERROR",
            status_message=str(event.error),
            metadata={"step": event.step},
        )
        self._root_span.end()
        self._lf.flush()
        self._trace_id = None
        self._root_span = None

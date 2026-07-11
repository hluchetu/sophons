from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from sophons import Message
from sophons.agents import Agent
from sophons.documents import Document
from sophons.observability import SophonsTelemetry, _semconv
from sophons.retrieval import BM25Retriever
from sophons.splitters import RecursiveCharacterSplitter
from sophons.tools import tool

# The global tracer provider can only be set once per process, so one
# provider + in-memory exporter is shared by every test in this module.
_EXPORTER = InMemorySpanExporter()
_PROVIDER = TracerProvider()
_PROVIDER.add_span_processor(SimpleSpanProcessor(_EXPORTER))
trace.set_tracer_provider(_PROVIDER)


@pytest.fixture()
def exporter() -> InMemorySpanExporter:
    _EXPORTER.clear()
    return _EXPORTER


def _spans_by_name(exporter: InMemorySpanExporter) -> dict[str, list]:
    spans: dict[str, list] = {}
    for span in exporter.get_finished_spans():
        spans.setdefault(span.name, []).append(span)
    return spans


# ── Components ─────────────────────────────────────────────────────────────────


def test_splitter_emits_span_with_counts(exporter: InMemorySpanExporter) -> None:
    splitter = RecursiveCharacterSplitter(chunk_size=10, chunk_overlap=0)
    document = Document(id="doc_1", content="one two three four five six seven")

    chunks = splitter.split_documents([document])

    (span,) = exporter.get_finished_spans()
    assert span.name == "splitter.split"
    assert span.attributes[_semconv.SPLITTER] == "recursive"
    assert span.attributes[_semconv.DOCUMENT_COUNT] == 1
    assert span.attributes[_semconv.CHUNK_COUNT] == len(chunks)


def test_retriever_emits_span_with_result_count(
    exporter: InMemorySpanExporter,
) -> None:
    retriever = BM25Retriever(
        [
            Document(id="1", content="the quick brown fox"),
            Document(id="2", content="pack my box with jugs"),
        ]
    )

    results = retriever.retrieve("quick fox", limit=1)

    (span,) = exporter.get_finished_spans()
    assert span.name == "retriever.search"
    assert span.attributes[_semconv.RETRIEVER] == "bm25"
    assert span.attributes[_semconv.LIMIT] == 1
    assert span.attributes[_semconv.RESULT_COUNT] == len(results)


# ── Agent loop ─────────────────────────────────────────────────────────────────


class ToolCallingModel:
    """First call requests the `add` tool, second call answers."""

    def __init__(self) -> None:
        self.calls = 0

    def invoke(self, messages: list[Message], tools: list | None = None) -> Message:
        self.calls += 1
        if self.calls == 1:
            return Message(
                role="assistant",
                content="",
                metadata={
                    "usage": {"input_tokens": 10, "output_tokens": 4},
                    "tool_calls": [
                        {
                            "tool_use_id": "call_1",
                            "name": "add",
                            "input": {"a": 2, "b": 3},
                        }
                    ],
                },
            )
        return Message(
            role="assistant",
            content="The answer is 5.",
            metadata={"usage": {"input_tokens": 20, "output_tokens": 6}},
        )


def test_agent_run_produces_span_tree(exporter: InMemorySpanExporter) -> None:
    @tool
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    agent = Agent(model=ToolCallingModel(), tools=[add])
    result = agent.run_sync("What is 2 + 3?", session_id="session-42")
    assert result.success is True

    spans = _spans_by_name(exporter)
    assert set(spans) == {"invoke_agent", "chat", "execute_tool add"}

    (root,) = spans["invoke_agent"]
    assert root.attributes[_semconv.SESSION_ID] == "session-42"
    assert root.attributes[_semconv.STOP_REASON] == "end_turn"
    assert root.attributes[_semconv.MODEL_CALLS] == 2
    assert root.attributes[_semconv.TOOL_CALLS] == 1
    assert root.attributes[_semconv.INPUT_TOKENS] == 30
    assert root.attributes[_semconv.OUTPUT_TOKENS] == 10

    chat_spans = spans["chat"]
    assert len(chat_spans) == 2
    for chat in chat_spans:
        assert chat.parent.span_id == root.context.span_id
    assert chat_spans[0].attributes[_semconv.INPUT_TOKENS] == 10

    (tool_span,) = spans["execute_tool add"]
    assert tool_span.parent.span_id == root.context.span_id
    assert tool_span.attributes[_semconv.TOOL_NAME] == "add"
    assert tool_span.attributes[_semconv.TOOL_CALL_ID] == "call_1"


def test_failing_tool_marks_span_as_error(exporter: InMemorySpanExporter) -> None:
    @tool
    def explode() -> str:
        """Always fails."""
        raise RuntimeError("boom")

    class ExplodingToolModel:
        def __init__(self) -> None:
            self.calls = 0

        def invoke(self, messages, tools=None) -> Message:
            self.calls += 1
            if self.calls == 1:
                return Message(
                    role="assistant",
                    content="",
                    metadata={
                        "tool_calls": [
                            {"tool_use_id": "call_1", "name": "explode", "input": {}}
                        ]
                    },
                )
            return Message(role="assistant", content="It failed.")

    agent = Agent(model=ExplodingToolModel(), tools=[explode])
    agent.run_sync("Try the tool.")

    spans = _spans_by_name(exporter)
    (tool_span,) = spans["execute_tool explode"]
    assert not tool_span.status.is_ok
    assert "boom" in tool_span.status.description


def test_component_span_nests_under_tool_span(
    exporter: InMemorySpanExporter,
) -> None:
    retriever = BM25Retriever([Document(id="1", content="paris is in france")])

    @tool
    def search(query: str) -> str:
        """Search documents."""
        results = retriever.retrieve(query, limit=1)
        return results[0].content if results else "nothing"

    class SearchingModel:
        def __init__(self) -> None:
            self.calls = 0

        def invoke(self, messages, tools=None) -> Message:
            self.calls += 1
            if self.calls == 1:
                return Message(
                    role="assistant",
                    content="",
                    metadata={
                        "tool_calls": [
                            {
                                "tool_use_id": "call_1",
                                "name": "search",
                                "input": {"query": "paris"},
                            }
                        ]
                    },
                )
            return Message(role="assistant", content="Paris is in France.")

    agent = Agent(model=SearchingModel(), tools=[search])
    agent.run_sync("Where is Paris?")

    spans = _spans_by_name(exporter)
    (tool_span,) = spans["execute_tool search"]
    (retriever_span,) = spans["retriever.search"]
    assert retriever_span.parent.span_id == tool_span.context.span_id


# ── SophonsTelemetry ───────────────────────────────────────────────────────────


def test_telemetry_accepts_existing_provider_and_chains() -> None:
    provider = TracerProvider()
    telemetry = SophonsTelemetry(tracer_provider=provider)

    assert telemetry.tracer_provider is provider
    assert telemetry.setup_console_exporter() is telemetry


def test_components_work_without_configured_sdk() -> None:
    """With bare opentelemetry-api (no provider), spans no-op silently."""
    code = (
        "from sophons.documents import Document\n"
        "from sophons.splitters import RecursiveCharacterSplitter\n"
        "splitter = RecursiveCharacterSplitter(chunk_size=10, chunk_overlap=0)\n"
        "chunks = splitter.split_documents([Document(id='d', content='hello world')])\n"
        "assert chunks\n"
        "print('ok')\n"
    )
    src = Path(__file__).resolve().parent.parent / "src"
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(src)},
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"

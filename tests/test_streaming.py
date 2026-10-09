"""Model streaming: events, the accumulator, providers and Agent.stream()."""
import asyncio
import io
import json
import threading
import types

import pytest

from sophons.agents import Agent, AfterToolCall, BeforeToolCall, ModelStreamed
from sophons.agents.responses import AgentResult
from sophons.errors import ModelResponseError, ModelThrottledError, ModelTimeoutError
from sophons.integrations.models import deepseek as deepseek_module
from sophons.integrations.models import ollama as ollama_module
from sophons.models import (
    Message,
    MessageComplete,
    MessageStop,
    ReasoningDelta,
    StreamingChatModel,
    TextDelta,
    ToolCallDelta,
    UsageUpdate,
    aprocess_stream,
    collect,
    events_from_message,
    process_stream,
    stream_model,
)
from sophons.tools import tool

EVENTS = [
    ReasoningDelta("thinking"),
    TextDelta("The tenant "),
    TextDelta("cannot sublet."),
    ToolCallDelta(index=0, tool_use_id="call-1", name="lookup", arguments='{"clause"'),
    ToolCallDelta(index=1, tool_use_id="call-2", name="echo", arguments=""),
    ToolCallDelta(index=0, arguments=': "10"}'),
    UsageUpdate(input_tokens=12, output_tokens=7),
    MessageStop("tool_use"),
]


def test_accumulator_assembles_text_tool_calls_usage_and_stop_reason():
    seen = list(process_stream(iter(EVENTS)))
    assert seen[:-1] == EVENTS  # every event is passed through unchanged, in order
    final = seen[-1]
    assert isinstance(final, MessageComplete)
    assert final.message.content == "The tenant cannot sublet."
    assert final.message.metadata["tool_calls"] == [
        {"tool_use_id": "call-1", "name": "lookup", "input": {"clause": "10"}},
        {"tool_use_id": "call-2", "name": "echo", "input": {}},  # no arguments means none
    ]
    assert final.message.metadata["reasoning"] == "thinking"
    assert final.usage == {"input_tokens": 12, "output_tokens": 7} == final.message.metadata["usage"]
    assert final.stop_reason == "tool_use" and final.time_to_first_token_ms >= 0
    assert collect(iter(EVENTS)) == final.message
    plain = list(process_stream(iter([TextDelta("Hi")])))[-1]
    assert (plain.stop_reason, plain.message.metadata) == ("end_turn", {})
    # A tool call with no explicit stop still reports that tools were requested.
    assert list(process_stream(iter(EVENTS[3:6])))[-1].stop_reason == "tool_use"


def test_cut_off_tool_arguments_are_an_error_not_an_empty_call():
    broken = [ToolCallDelta(index=0, tool_use_id="c", name="answer", arguments='{"claims": ['),
              MessageStop("max_tokens")]
    with pytest.raises(ModelResponseError) as error:
        list(process_stream(iter(broken)))
    assert error.value.details == {"tool": "answer", "stop_reason": "max_tokens"}


def test_cancellation_stops_between_events_and_keeps_what_arrived():
    signal = threading.Event()
    closed = []

    def source():
        try:
            yield TextDelta("first ")
            yield TextDelta("second")
            yield ToolCallDelta(index=0, name="answer", arguments='{"unfinished')
        finally:
            closed.append(True)

    seen = []
    for event in process_stream(source(), cancel_signal=signal):
        seen.append(event)
        signal.set()  # the caller cancels after the first piece arrives
    assert seen[0] == TextDelta("first ") and len(seen) == 2
    assert (seen[1].stop_reason, seen[1].message.content) == ("cancelled", "first ")
    assert closed == [True]  # the provider stream is released


def test_async_stream_is_accumulated_and_a_silent_stream_times_out():
    async def source(delay=0.0):
        yield TextDelta("a")
        await asyncio.sleep(delay)
        yield MessageStop()

    async def run():
        seen = [event async for event in aprocess_stream(source())]
        assert seen[-1].message.content == "a" and seen[-1].stop_reason == "end_turn"
        with pytest.raises(ModelTimeoutError, match="inactivity"):
            [event async for event in aprocess_stream(source(delay=0.2), inactivity_timeout=0.02)]
        signal = threading.Event()
        signal.set()
        cancelled = [event async for event in aprocess_stream(source(), cancel_signal=signal)]
        assert [type(e).__name__ for e in cancelled] == ["MessageComplete"]
        assert cancelled[0].stop_reason == "cancelled"

    asyncio.run(run())


def test_a_model_without_stream_is_replayed_from_invoke():
    message = Message(role="assistant", content="Hello", metadata={
        "tool_calls": [{"tool_use_id": "c", "name": "echo", "input": {"value": "x"}}],
        "usage": {"input_tokens": 3, "output_tokens": 1},
    })

    class InvokeOnly:
        def invoke(self, messages, tools=None):
            return message

    assert not isinstance(InvokeOnly(), StreamingChatModel)
    assert collect(stream_model(InvokeOnly(), [])) == message
    assert collect(events_from_message(message)) == message
    assert list(events_from_message(Message(role="assistant", content="Hi")))[-1] == MessageStop("end_turn")


# --- providers ---------------------------------------------------------------


def chunk(*, content=None, reasoning=None, tool=None, finish=None, usage=None):
    delta = types.SimpleNamespace(content=content, reasoning_content=reasoning,
                                  tool_calls=[tool] if tool else None)
    choices = [] if usage and not (content or tool or finish) else [
        types.SimpleNamespace(delta=delta, finish_reason=finish)]
    return types.SimpleNamespace(choices=choices, usage=usage)


def tool_fragment(index, arguments, *, id=None, name=None):
    return types.SimpleNamespace(index=index, id=id,
                                 function=types.SimpleNamespace(name=name, arguments=arguments))


def deepseek(chunks, *, fail=None, requests=None):
    class Stream(list):
        closed = False

        def close(self):
            type(self).closed = True

    stream = Stream(chunks)

    def create(**kwargs):
        if requests is not None:
            requests.append(kwargs)
        if fail:
            raise fail
        return iter_with_close(stream)

    model = deepseek_module.DeepSeekModel.__new__(deepseek_module.DeepSeekModel)
    model.model, model._thinking = "deepseek-test", False
    model._adapter = deepseek_module.OpenAICompatAdapter()
    model._client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create)))
    return model, stream


class iter_with_close:
    def __init__(self, items):
        self._items, self._iterator = items, iter(items)

    def __iter__(self):
        return self

    def __next__(self):
        item = next(self._iterator)
        if isinstance(item, Exception):
            raise item
        return item

    def close(self):
        self._items.close()


def test_deepseek_chunks_become_events_that_assemble_to_one_message():
    requests = []
    usage = types.SimpleNamespace(prompt_tokens=20, completion_tokens=5, prompt_cache_hit_tokens=4)
    model, stream = deepseek([
        chunk(reasoning="hmm"),
        chunk(content="No. "),
        chunk(content="Clause 10."),
        chunk(tool=tool_fragment(0, '{"a":', id="call-9", name="cite")),
        chunk(tool=tool_fragment(0, " 1}")),
        chunk(finish="tool_calls"),
        chunk(usage=usage),
    ], requests=requests)
    assert isinstance(model, StreamingChatModel)
    events = list(model.stream([Message(role="user", content="Can the tenant sublet?")]))
    assert events == [
        ReasoningDelta("hmm"), TextDelta("No. "), TextDelta("Clause 10."),
        ToolCallDelta(index=0, tool_use_id="call-9", name="cite", arguments='{"a":'),
        ToolCallDelta(index=0, arguments=" 1}"),
        MessageStop("tool_use"),
        UsageUpdate(input_tokens=20, output_tokens=5, cache_read_tokens=4),
    ]
    assert requests[0]["stream"] is True and requests[0]["stream_options"] == {"include_usage": True}
    assert "tools" not in requests[0] and type(stream).closed
    message = collect(iter(events))
    assert message.content == "No. Clause 10."
    assert message.metadata["tool_calls"] == [{"tool_use_id": "call-9", "name": "cite", "input": {"a": 1}}]
    assert message.metadata["usage"] == {"input_tokens": 20, "output_tokens": 5, "cache_read_tokens": 4}


def test_deepseek_stream_failures_are_sophons_model_errors():
    class RateLimitError(Exception):
        status_code = 429

    class APITimeoutError(Exception):
        pass

    refused, _ = deepseek([], fail=RateLimitError())
    with pytest.raises(ModelThrottledError):
        list(refused.stream([]))
    stalled, stream = deepseek([chunk(content="partial"), APITimeoutError()])
    with pytest.raises(ModelTimeoutError):
        list(stalled.stream([]))
    assert type(stream).closed  # the request is released even when the stream fails
    truncated, _ = deepseek([chunk(content="no finish reason")])
    with pytest.raises(ModelResponseError, match="without a finish reason"):
        list(truncated.stream([]))
    stopped = list(deepseek([chunk(content="x", finish="length")])[0].stream([]))
    assert stopped[-1] == MessageStop("max_tokens")


def test_ollama_lines_become_events(monkeypatch):
    lines = [
        json.dumps({"message": {"content": "No. "}, "done": False}),
        "",
        json.dumps({"message": {"content": "", "tool_calls": [
            {"function": {"name": "cite", "arguments": {"clause": 10}}}]}, "done": False}),
        json.dumps({"message": {"content": ""}, "done": True, "done_reason": "stop",
                    "prompt_eval_count": 9, "eval_count": 3}),
    ]
    sent = {}

    class Response(io.BytesIO):
        pass

    def urlopen(request, timeout):
        sent["body"], sent["timeout"] = json.loads(request.data), timeout
        return Response("\n".join(lines).encode())

    monkeypatch.setattr(ollama_module.urllib.request, "urlopen", urlopen)
    model = ollama_module.OllamaModel("llama-test")
    events = list(model.stream([Message(role="user", content="Q")]))
    assert events == [
        TextDelta("No. "),
        ToolCallDelta(index=0, tool_use_id=None, name="cite", arguments='{"clause": 10}'),
        UsageUpdate(input_tokens=9, output_tokens=3),
        MessageStop("tool_use"),
    ]
    assert sent["body"]["stream"] is True
    assert collect(iter(events)).metadata["tool_calls"][0]["input"] == {"clause": 10}

    def unreachable(request, timeout):
        raise OSError("refused")

    monkeypatch.setattr(ollama_module.urllib.request, "urlopen", unreachable)
    with pytest.raises(Exception, match="Could not connect to Ollama"):
        list(model.stream([]))
    with pytest.raises(ModelResponseError, match="before it was done"):
        list(ollama_module._stream_events([json.dumps({"message": {"content": "x"}, "done": False})], model="m"))
    with pytest.raises(ModelResponseError, match="not valid JSON"):
        list(ollama_module._stream_events(["{nope"], model="m"))


# --- agent -------------------------------------------------------------------


class StreamingModel:
    """Asks for one tool call, then answers; counts which path the loop used."""

    def __init__(self, name="x"):
        self.name, self.calls, self.invoked, self.streamed = name, 0, 0, 0

    def _turn(self):
        self.calls += 1
        if self.calls == 1:
            return [TextDelta(f"Checking {self.name}. "),
                    ToolCallDelta(index=0, tool_use_id="t1", name="echo", arguments='{"value": '),
                    ToolCallDelta(index=0, arguments=f'"{self.name}"' + "}"),
                    MessageStop("tool_use")]
        return [TextDelta("Done "), TextDelta(self.name), UsageUpdate(input_tokens=5, output_tokens=2),
                MessageStop("end_turn")]

    def stream(self, messages, tools=None):
        self.streamed += 1
        yield from self._turn()

    def invoke(self, messages, tools=None):
        self.invoked += 1
        return collect(iter(self._turn()))


def echo_tool(seen):
    @tool
    def echo(value: str) -> str:
        """Echo a value."""
        seen.append(value)
        return value.upper()

    return echo


def test_agent_stream_yields_model_events_tool_events_and_the_result():
    async def run():
        executed = []
        model = StreamingModel()
        agent = Agent(model=model, tools=[echo_tool(executed)])
        events = [event async for event in agent.stream("Go")]
        result = events[-1]
        assert isinstance(result, AgentResult) and result.success
        assert result.message == "Done x" and executed == ["x"]
        kinds = [type(event).__name__ for event in events[:-1]]
        assert kinds == [
            "TextDelta", "ToolCallDelta", "ToolCallDelta", "MessageStop",
            "BeforeToolCall", "AfterToolCall",
            "TextDelta", "TextDelta", "UsageUpdate", "MessageStop",
        ]
        assert "".join(e.text for e in events if isinstance(e, TextDelta)) == "Checking x. Done x"
        after = next(e for e in events if isinstance(e, AfterToolCall))
        assert json.loads(after.tool_result.content) == {"result": "X"}
        assert isinstance(next(e for e in events if isinstance(e, BeforeToolCall)), BeforeToolCall)
        assert (model.streamed, model.invoked) == (2, 0)

    asyncio.run(run())


def test_runs_do_not_stream_unless_something_listens():
    model = StreamingModel()
    result = Agent(model=model, tools=[echo_tool([])]).run_sync("Go")
    assert result.message == "Done x" and (model.streamed, model.invoked) == (0, 2)

    hooked, heard = StreamingModel(), []
    agent = Agent(model=hooked, tools=[echo_tool([])])
    agent._hooks.register(ModelStreamed, heard.append)
    assert agent.run_sync("Go").message == "Done x"
    assert (hooked.streamed, hooked.invoked) == (2, 0)
    assert [type(e.event).__name__ for e in heard][:2] == ["TextDelta", "ToolCallDelta"]
    assert {e.step for e in heard} == {0, 1}  # the same step numbering as BeforeModelCall


def test_concurrent_streams_on_one_agent_receive_only_their_own_events():
    class Shared:
        def stream(self, messages, tools=None):
            name = messages[-1].content
            for piece in (name, "-", name):
                yield TextDelta(piece)
            yield MessageStop()

        def invoke(self, messages, tools=None):
            raise AssertionError("streamed runs must not fall back to invoke")

    async def run():
        agent = Agent(model=Shared())

        async def texts(prompt):
            return "".join([e.text async for e in agent.stream(prompt) if isinstance(e, TextDelta)])

        assert await asyncio.gather(texts("a"), texts("b"), texts("c")) == ["a-a", "b-b", "c-c"]

    asyncio.run(run())


def test_invoke_only_models_stream_as_one_delta_and_early_exit_cancels():
    class InvokeOnly:
        def invoke(self, messages, tools=None):
            return Message(role="assistant", content="Whole answer")

    async def run():
        events = [event async for event in Agent(model=InvokeOnly()).stream("Go")]
        assert [type(e).__name__ for e in events] == ["TextDelta", "MessageStop", "AgentResult"]
        assert events[0].text == "Whole answer"

        stream = Agent(model=StreamingModel(), tools=[echo_tool([])]).stream("Go")
        first = await stream.__anext__()
        assert isinstance(first, TextDelta)
        await stream.aclose()  # leaving early must not hang or raise

    asyncio.run(run())

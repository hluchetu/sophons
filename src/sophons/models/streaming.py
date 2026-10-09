from __future__ import annotations

import asyncio
import json
import threading
import time
from collections.abc import AsyncIterable, AsyncIterator, Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

from sophons.errors import ModelResponseError, ModelTimeoutError
from sophons.models.messages import Message


@dataclass(frozen=True, slots=True)
class TextDelta:
    """A piece of the visible answer."""

    text: str


@dataclass(frozen=True, slots=True)
class ReasoningDelta:
    """A piece of provider reasoning, when the provider exposes it."""

    text: str


@dataclass(frozen=True, slots=True)
class ToolCallDelta:
    """Part of one tool call. ``arguments`` is a fragment of its JSON arguments.

    ``index`` says which call the fragment belongs to; the ID and name arrive
    with the first fragment and need not be repeated.
    """

    index: int
    tool_use_id: str | None = None
    name: str | None = None
    arguments: str = ""


@dataclass(frozen=True, slots=True)
class UsageUpdate:
    input_tokens: int | None = None
    output_tokens: int | None = None
    cache_read_tokens: int | None = None


@dataclass(frozen=True, slots=True)
class MessageStop:
    """The provider finished: end_turn, tool_use, max_tokens or content_filtered."""

    stop_reason: str = "end_turn"


StreamEvent = TextDelta | ReasoningDelta | ToolCallDelta | UsageUpdate | MessageStop


@dataclass(frozen=True, slots=True)
class MessageComplete:
    """Emitted once by the accumulator, after the last provider event."""

    message: Message
    stop_reason: str
    usage: dict[str, int] = field(default_factory=dict)
    time_to_first_token_ms: float | None = None


class StreamAccumulator:
    """Builds the final message from stream events.

    Tool-call arguments arrive as text fragments and are parsed only once the
    stream ends. Arguments that are not valid JSON raise ``ModelResponseError``
    instead of becoming an empty call: a structured answer cut off part-way must
    not be mistaken for an empty one.
    """

    def __init__(self) -> None:
        self._text: list[str] = []
        self._reasoning: list[str] = []
        self._tools: dict[int, dict[str, Any]] = {}
        self._usage: dict[str, int] = {}
        self._stop_reason: str | None = None
        self._started = time.monotonic()
        self._first: float | None = None

    def feed(self, event: StreamEvent) -> None:
        if isinstance(event, (TextDelta, ReasoningDelta, ToolCallDelta)) and self._first is None:
            self._first = time.monotonic()
        if isinstance(event, TextDelta):
            self._text.append(event.text)
        elif isinstance(event, ReasoningDelta):
            self._reasoning.append(event.text)
        elif isinstance(event, ToolCallDelta):
            call = self._tools.setdefault(event.index, {"tool_use_id": None, "name": None, "arguments": []})
            call["tool_use_id"] = call["tool_use_id"] or event.tool_use_id
            call["name"] = call["name"] or event.name
            call["arguments"].append(event.arguments)
        elif isinstance(event, UsageUpdate):
            for key in ("input_tokens", "output_tokens", "cache_read_tokens"):
                value = getattr(event, key)
                if value is not None:
                    self._usage[key] = value
        elif isinstance(event, MessageStop):
            self._stop_reason = event.stop_reason

    def complete(self, *, stop_reason: str | None = None, strict: bool = True) -> MessageComplete:
        """The assembled message. ``strict=False`` keeps what arrived of a cut-off call."""

        tool_calls = []
        for index in sorted(self._tools):
            call = self._tools[index]
            if not call["name"]:
                continue
            raw = "".join(call["arguments"])
            try:
                arguments = json.loads(raw) if raw.strip() else {}
            except json.JSONDecodeError as error:
                if strict:
                    raise ModelResponseError(
                        "The model's tool-call arguments were not valid JSON.",
                        details={"tool": call["name"], "stop_reason": self._stop_reason},
                    ) from error
                arguments = {"raw": raw}
            tool_calls.append(
                {"tool_use_id": call["tool_use_id"], "name": call["name"], "input": arguments}
            )
        metadata: dict[str, Any] = {}
        if tool_calls:
            metadata["tool_calls"] = tool_calls
        if self._reasoning:
            metadata["reasoning"] = "".join(self._reasoning)
        if self._usage:
            metadata["usage"] = dict(self._usage)
        reason = stop_reason or self._stop_reason or "end_turn"
        if tool_calls and reason == "end_turn":
            reason = "tool_use"
        first = None if self._first is None else (self._first - self._started) * 1000
        return MessageComplete(
            message=Message(role="assistant", content="".join(self._text), metadata=metadata),
            stop_reason=reason,
            usage=dict(self._usage),
            time_to_first_token_ms=first,
        )


def process_stream(
    events: Iterable[StreamEvent], *, cancel_signal: threading.Event | None = None
) -> Iterator[StreamEvent | MessageComplete]:
    """Pass events through, then yield one ``MessageComplete``.

    The cancel signal is checked before each event. A cancelled stream ends with
    a completion whose stop reason is ``cancelled`` and whose message holds what
    had arrived.
    """

    accumulator = StreamAccumulator()
    iterator = iter(events)
    try:
        while True:
            if cancel_signal is not None and cancel_signal.is_set():
                yield accumulator.complete(stop_reason="cancelled", strict=False)
                return
            try:
                event = next(iterator)
            except StopIteration:
                break
            accumulator.feed(event)
            yield event
    finally:
        close = getattr(iterator, "close", None)
        if close is not None:
            close()  # release the provider connection if the consumer stops early
    yield accumulator.complete()


async def aprocess_stream(
    events: AsyncIterable[StreamEvent],
    *,
    cancel_signal: threading.Event | None = None,
    inactivity_timeout: float | None = None,
) -> AsyncIterator[StreamEvent | MessageComplete]:
    """Async ``process_stream`` that also ends a stream which goes silent."""

    accumulator = StreamAccumulator()
    iterator = events.__aiter__()
    try:
        while True:
            if cancel_signal is not None and cancel_signal.is_set():
                yield accumulator.complete(stop_reason="cancelled", strict=False)
                return
            try:
                event = await asyncio.wait_for(iterator.__anext__(), inactivity_timeout)
            except StopAsyncIteration:
                break
            except asyncio.TimeoutError as error:
                raise ModelTimeoutError(
                    "The model stream produced nothing within the inactivity limit.",
                    details={"inactivity_timeout_seconds": inactivity_timeout},
                ) from error
            accumulator.feed(event)
            yield event
    finally:
        close = getattr(iterator, "aclose", None)
        if close is not None:
            await close()
    yield accumulator.complete()


def events_from_message(message: Message, *, stop_reason: str | None = None) -> Iterator[StreamEvent]:
    """Replay a finished message as stream events, for models without ``stream()``."""

    if message.metadata.get("reasoning"):
        yield ReasoningDelta(str(message.metadata["reasoning"]))
    if message.content:
        yield TextDelta(message.content)
    calls = message.metadata.get("tool_calls") or []
    for index, call in enumerate(calls):
        yield ToolCallDelta(
            index=index,
            tool_use_id=call.get("tool_use_id"),
            name=call.get("name"),
            arguments=json.dumps(call.get("input") or {}),
        )
    usage = message.metadata.get("usage") or {}
    if usage:
        yield UsageUpdate(
            input_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens"),
            cache_read_tokens=usage.get("cache_read_tokens"),
        )
    yield MessageStop(stop_reason or ("tool_use" if calls else "end_turn"))


def stream_model(model: Any, messages: list[Message], tools: list | None = None) -> Iterator[StreamEvent]:
    """Stream any sync chat model: its own ``stream()``, or ``invoke()`` replayed."""

    stream = getattr(model, "stream", None)
    if callable(stream):
        yield from stream(messages, tools=tools)
        return
    yield from events_from_message(model.invoke(messages, tools=tools))


def collect(events: Iterable[StreamEvent]) -> Message:
    """Read a stream to its end and return the assembled message."""

    final = None
    for event in process_stream(events):
        final = event
    assert isinstance(final, MessageComplete)
    return final.message

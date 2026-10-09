"""Delivery of run events to whoever is consuming ``Agent.stream()``.

The consumer is held in a context variable, not on the agent, so two streams
running concurrently on one agent each receive only their own events. The
variable is copied into the worker thread that reads a sync model stream.
"""
from __future__ import annotations

import asyncio
import contextvars
from collections.abc import Callable
from typing import Any

_sink: contextvars.ContextVar[Callable[[Any], None] | None] = contextvars.ContextVar(
    "sophons_stream_sink", default=None
)


def is_streaming() -> bool:
    """True while an ``Agent.stream()`` is consuming the current run."""

    return _sink.get() is not None


def emit(event: Any) -> None:
    """Hand an event to the current stream consumer, if there is one."""

    sink = _sink.get()
    if sink is not None:
        sink(event)


def open_sink(queue: asyncio.Queue) -> contextvars.Token:
    """Route events from the current context, and threads copied from it, to a queue."""

    loop = asyncio.get_running_loop()

    def push(event: Any) -> None:
        # Stream events arrive from a worker thread; the queue belongs to the loop.
        loop.call_soon_threadsafe(queue.put_nowait, event)

    return _sink.set(push)


def close_sink(token: contextvars.Token) -> None:
    _sink.reset(token)

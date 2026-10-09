from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from sophons.models.messages import Message

if TYPE_CHECKING:
    from sophons.models.streaming import StreamEvent


@runtime_checkable
class ChatModel(Protocol):
    """Sync chat model contract: messages in, assistant message out."""

    def invoke(self, messages: list[Message], tools: list | None = None) -> Message:
        ...


@runtime_checkable
class AsyncChatModel(Protocol):
    """Async chat model contract: messages in, assistant message out."""

    async def invoke(self, messages: list[Message], tools: list | None = None) -> Message:
        ...


@runtime_checkable
class StreamingChatModel(Protocol):
    """Sync streaming contract: messages in, provider-neutral stream events out.

    The events must assemble to the message ``invoke`` would return. Provider
    failures, including a stream that stalls, are raised as Sophons model errors.
    """

    def stream(self, messages: list[Message], tools: list | None = None) -> Iterator[StreamEvent]:
        ...


@runtime_checkable
class AsyncStreamingChatModel(Protocol):
    """Async streaming contract."""

    def stream(self, messages: list[Message], tools: list | None = None) -> AsyncIterator[StreamEvent]:
        ...

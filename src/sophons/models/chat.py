from __future__ import annotations

from typing import Protocol, runtime_checkable

from sophons.models.messages import Message


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

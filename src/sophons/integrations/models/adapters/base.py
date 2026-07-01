from __future__ import annotations

from typing import Protocol

from sophons.models.messages import Message
from sophons.tools.base import Tool


class ProviderAdapter(Protocol):
    """Contract every provider adapter must satisfy."""

    def serialize_messages(self, messages: list[Message]) -> list[dict]: ...
    def serialize_tools(self, tools: list[Tool]) -> list[dict]: ...

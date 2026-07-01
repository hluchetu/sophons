from __future__ import annotations

from typing import Any

from sophons.integrations.models.adapters.anthropic import AnthropicAdapter
from sophons.integrations.models.settings import ModelSettings
from sophons.models.messages import Message
from sophons.tools.base import Tool


class AnthropicClient:
    def __init__(
        self,
        model: str,
        api_key: str,
        tools: list[Tool] | None = None,
    ) -> None:
        try:
            import anthropic as _sdk
        except ImportError as e:
            raise ImportError("pip install anthropic") from e

        self.model = model
        self._tools = tools or []
        self._adapter = AnthropicAdapter()
        self._client = _sdk.Anthropic(api_key=api_key)

    def invoke(self, messages: list[Message]) -> Message:
        system = next((m.content for m in messages if m.role == "system"), None)
        non_system = [m for m in messages if m.role != "system"]

        kwargs: dict[str, Any] = dict(
            model=self.model,
            max_tokens=4096,
            messages=self._adapter.serialize_messages(non_system),
        )
        if system:
            kwargs["system"] = system
        if self._tools:
            kwargs["tools"] = self._adapter.serialize_tools(self._tools)

        response = self._client.messages.create(**kwargs)

        text = " ".join(b.text for b in response.content if b.type == "text")
        tool_calls = [
            {"tool_use_id": b.id, "name": b.name, "input": b.input}
            for b in response.content
            if b.type == "tool_use"
        ]
        metadata: dict[str, Any] = {"tool_calls": tool_calls} if tool_calls else {}

        return Message(role="assistant", content=text, metadata=metadata)


class AnthropicProvider:
    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    def get_model(
        self,
        model_name: str,
        settings: ModelSettings,
        tools: list[Tool] | None = None,
    ) -> AnthropicClient:
        return AnthropicClient(model=model_name, api_key=self._api_key, tools=tools)

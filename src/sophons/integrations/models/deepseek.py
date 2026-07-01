from __future__ import annotations

import json

from openai import OpenAI

from sophons.integrations.models.adapters.openai_compat import OpenAICompatAdapter
from sophons.integrations.models.settings import ModelSettings
from sophons.models.messages import Message
from sophons.tools.base import Tool


class DeepSeekClient:
    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str = "https://api.deepseek.com/v1",
        thinking: bool = False,
        tools: list[Tool] | None = None,
    ) -> None:
        self.model = model
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._thinking = thinking
        self._tools = tools or []
        self._adapter = OpenAICompatAdapter()

    def invoke(self, messages: list[Message]) -> Message:
        kwargs: dict = dict(
            model=self.model,
            messages=self._adapter.serialize_messages(messages),
            extra_body={
                "thinking": {"type": "enabled" if self._thinking else "disabled"}
            },
            temperature=0,
            stream=False,
        )
        if self._tools:
            kwargs["tools"] = self._adapter.serialize_tools(self._tools)

        response = self._client.chat.completions.create(**kwargs)
        message = response.choices[0].message

        tool_calls = _normalize_tool_calls(getattr(message, "tool_calls", None) or [])
        metadata = {"tool_calls": tool_calls} if tool_calls else {}

        return Message(
            role="assistant", content=message.content or "", metadata=metadata
        )


class DeepSeekProvider:
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.deepseek.com/v1",
        thinking: bool = False,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url
        self._thinking = thinking

    def get_model(
        self,
        model_name: str,
        settings: ModelSettings,
        tools: list[Tool] | None = None,
    ) -> DeepSeekClient:
        return DeepSeekClient(
            model=model_name,
            api_key=self._api_key,
            base_url=self._base_url,
            thinking=self._thinking,
            tools=tools,
        )


def _normalize_tool_calls(raw: list) -> list[dict]:
    result = []
    for item in raw:
        fn = getattr(item, "function", None)
        name = getattr(fn, "name", None)
        if not name:
            continue
        args = getattr(fn, "arguments", {}) or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {"raw": args}
        result.append(
            {"tool_use_id": getattr(item, "id", None), "name": name, "input": args}
        )
    return result

from __future__ import annotations

import json

from openai import OpenAI

from sophons.integrations.models.adapters.openai_compat import OpenAICompatAdapter
from sophons.models.messages import Message
from sophons.tools.base import Tool


class DeepSeekModel:
    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str = "https://api.deepseek.com/v1",
        thinking: bool = False,
    ) -> None:
        self.model = model
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._thinking = thinking
        self._adapter = OpenAICompatAdapter()

    def invoke(
        self, messages: list[Message], tools: list[Tool] | None = None
    ) -> Message:
        kwargs: dict = dict(
            model=self.model,
            messages=self._adapter.serialize_messages(messages),
            temperature=0,
            stream=False,
        )
        if self._thinking:
            kwargs["extra_body"] = {"thinking": {"type": "enabled"}}
        if tools:
            kwargs["tools"] = self._adapter.serialize_tools(tools)

        response = self._client.chat.completions.create(**kwargs)
        message = response.choices[0].message

        tool_calls = _normalize_tool_calls(getattr(message, "tool_calls", None) or [])
        metadata = {"tool_calls": tool_calls} if tool_calls else {}

        if getattr(message, "reasoning_content", None):
            metadata["reasoning"] = message.reasoning_content

        return Message(
            role="assistant", content=message.content or "", metadata=metadata
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

from __future__ import annotations

import json
import urllib.request
from typing import Any

from sophons.integrations.models.adapters.openai_compat import OpenAICompatAdapter
from sophons.integrations.models.settings import ModelSettings
from sophons.models.messages import Message
from sophons.tools.base import Tool


class OllamaModel:
    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        settings: ModelSettings | None = None,
    ) -> None:
        self.model = model
        self._base_url = base_url.rstrip("/")
        self._settings = settings or ModelSettings()
        self._adapter = OpenAICompatAdapter()

    def invoke(self, messages: list[Message], tools: list[Tool] | None = None) -> Message:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": self._adapter.serialize_messages(messages),
            "options": {
                "temperature": self._settings.temperature,
                "num_predict": self._settings.max_tokens,
            },
            "stream": False,
        }
        if tools:
            payload["tools"] = self._adapter.serialize_tools(tools)

        body = json.dumps(payload).encode()
        req = urllib.request.Request(
            f"{self._base_url}/api/chat",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self._settings.timeout_seconds) as resp:
            data = json.loads(resp.read().decode())

        message = data.get("message", {})
        tool_calls = _normalize_tool_calls(message.get("tool_calls") or [])
        metadata = {"tool_calls": tool_calls} if tool_calls else {}

        return Message(
            role="assistant",
            content=(message.get("content") or "").strip(),
            metadata=metadata,
        )



def _normalize_tool_calls(raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for item in raw:
        fn = item.get("function", item)
        name = fn.get("name")
        if not name:
            continue
        args = fn.get("arguments") or fn.get("input") or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {"raw": args}
        result.append({
            "tool_use_id": item.get("id") or item.get("tool_use_id"),
            "name": name,
            "input": args,
        })
    return result

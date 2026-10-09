from __future__ import annotations

import json
import urllib.request
from collections.abc import Iterator
from typing import Any

from sophons.errors import ModelError, ModelResponseError, ModelTimeoutError, ModelUnavailableError

from sophons.integrations.models.adapters.openai_compat import OpenAICompatAdapter
from sophons.integrations.models.settings import ModelSettings
from sophons.models.messages import Message
from sophons.models.streaming import MessageStop, StreamEvent, TextDelta, ToolCallDelta, UsageUpdate
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

    def stream(self, messages: list[Message], tools: list[Tool] | None = None) -> Iterator[StreamEvent]:
        """The same request as ``invoke``, read as Ollama's line-delimited stream."""
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": self._adapter.serialize_messages(messages),
            "options": {
                "temperature": self._settings.temperature,
                "num_predict": self._settings.max_tokens,
            },
            "stream": True,
        }
        if tools:
            payload["tools"] = self._adapter.serialize_tools(tools)
        request = urllib.request.Request(
            f"{self._base_url}/api/chat",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            response = urllib.request.urlopen(request, timeout=self._settings.timeout_seconds)
        except TimeoutError as exc:
            raise ModelTimeoutError("The Ollama request timed out.", details={"provider": "ollama"}) from exc
        except OSError as exc:
            raise ModelUnavailableError("Could not connect to Ollama.", details={"provider": "ollama"}) from exc
        with response:
            yield from _stream_events(response, model=self.model)



def _stream_events(lines: Any, *, model: str) -> Iterator[StreamEvent]:
    """Translate Ollama's one-JSON-object-per-line stream into Sophons events."""

    tool_index = 0
    finished = False
    try:
        for line in lines:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ModelResponseError(
                    "Ollama sent a stream line that was not valid JSON.",
                    details={"provider": "ollama", "model": model},
                ) from exc
            if data.get("error"):
                raise ModelError(
                    "The Ollama request failed.", details={"provider": "ollama", "model": model}
                )
            message = data.get("message") or {}
            if message.get("content"):
                yield TextDelta(message["content"])
            # Ollama sends each tool call whole, not as argument fragments.
            for call in _normalize_tool_calls(message.get("tool_calls") or []):
                yield ToolCallDelta(
                    index=tool_index,
                    tool_use_id=call["tool_use_id"],
                    name=call["name"],
                    arguments=json.dumps(call["input"]),
                )
                tool_index += 1
            if data.get("done"):
                finished = True
                if data.get("prompt_eval_count") is not None or data.get("eval_count") is not None:
                    yield UsageUpdate(
                        input_tokens=data.get("prompt_eval_count"),
                        output_tokens=data.get("eval_count"),
                    )
                reason = data.get("done_reason") or "stop"
                yield MessageStop(
                    "tool_use" if tool_index else {"stop": "end_turn", "length": "max_tokens"}.get(reason, reason)
                )
    except TimeoutError as exc:
        raise ModelTimeoutError(
            "The Ollama stream produced nothing within the timeout.",
            details={"provider": "ollama", "model": model},
        ) from exc
    if not finished:
        raise ModelResponseError(
            "Ollama ended the stream before it was done.",
            details={"provider": "ollama", "model": model},
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

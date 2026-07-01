from __future__ import annotations

import json

from sophons.models.messages import Message
from sophons.tools.base import Tool


class OpenAICompatAdapter:
    """Message serializer for OpenAI-compatible APIs (DeepSeek, Ollama, OpenAI).

    Handles the three message shapes the loop produces:
    - Plain user/system/assistant messages
    - Assistant messages that triggered tool calls (need tool_calls array)
    - Tool result messages (need tool_call_id to match back to the call)
    """

    def serialize_messages(self, messages: list[Message]) -> list[dict]:
        result = []
        for msg in messages:
            if msg.role == "tool":
                result.append({
                    "role": "tool",
                    "content": msg.content,
                    "tool_call_id": msg.metadata.get("tool_use_id", ""),
                })
            elif msg.role == "assistant" and msg.metadata.get("tool_calls"):
                result.append({
                    "role": "assistant",
                    "content": msg.content or None,
                    "tool_calls": [
                        {
                            "id": tc.get("tool_use_id", ""),
                            "type": "function",
                            "function": {
                                "name": tc.get("name", ""),
                                "arguments": json.dumps(tc.get("input", {})),
                            },
                        }
                        for tc in msg.metadata["tool_calls"]
                    ],
                })
            else:
                result.append({"role": msg.role, "content": msg.content})
        return result

    def serialize_tools(self, tools: list[Tool]) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.args_schema,
                },
            }
            for t in tools
        ]

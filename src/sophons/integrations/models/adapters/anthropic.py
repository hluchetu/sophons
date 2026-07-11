from __future__ import annotations

from sophons.models.messages import Message
from sophons.tools.base import Tool


class AnthropicAdapter:
    """Message serializer for the Anthropic Messages API.

    Key differences from OpenAI:
    - Tool results go back as role='user' with content blocks (not role='tool').
    - Consecutive tool results are merged into one user message.
    - Tool calls in assistant messages are content blocks, not a top-level field.
    - Tool definitions use 'input_schema' instead of 'parameters'.
    - System messages are passed as a separate API parameter, not in the message list.
    """

    def serialize_messages(self, messages: list[Message]) -> list[dict]:
        result: list[dict] = []
        i = 0
        while i < len(messages):
            msg = messages[i]

            if msg.role == "tool":
                # Merge consecutive tool results into a single user message
                tool_results = []
                while i < len(messages) and messages[i].role == "tool":
                    m = messages[i]
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": m.metadata.get("tool_use_id", ""),
                        "content": m.content,
                    })
                    i += 1
                result.append({"role": "user", "content": tool_results})
                continue

            if msg.role == "assistant":
                content: list[dict] = []
                if msg.content:
                    content.append({"type": "text", "text": msg.content})
                for tc in msg.metadata.get("tool_calls", []):
                    content.append({
                        "type": "tool_use",
                        "id": tc.get("tool_use_id", ""),
                        "name": tc.get("name", ""),
                        "input": tc.get("input", {}),
                    })
                result.append({"role": "assistant", "content": content})
                i += 1
                continue

            result.append({"role": msg.role, "content": msg.content})
            i += 1

        return result

    def serialize_tools(self, tools: list[Tool]) -> list[dict]:
        return [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.args_schema,
            }
            for t in tools
        ]

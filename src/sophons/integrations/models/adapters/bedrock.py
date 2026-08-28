from __future__ import annotations

from sophons.models.messages import Message
from sophons.tools.base import Tool


class BedrockAdapter:
    """Translate Sophons messages and tools to Bedrock Converse structures."""

    def serialize_messages(
        self,
        messages: list[Message],
    ) -> list[dict]:
        serialized: list[dict] = []
        index = 0

        while index < len(messages):
            message = messages[index]

            if message.role == "system":
                index += 1
                continue

            if message.role == "tool":
                tool_results: list[dict] = []

                while index < len(messages) and messages[index].role == "tool":
                    tool_message = messages[index]

                    tool_results.append(
                        {
                            "toolResult": {
                                "toolUseId": tool_message.metadata.get(
                                    "tool_use_id",
                                    "",
                                ),
                                "content": [
                                    {
                                        "text": tool_message.content,
                                    }
                                ],
                            }
                        }
                    )

                    index += 1

                serialized.append(
                    {
                        "role": "user",
                        "content": tool_results,
                    }
                )
                continue

            if message.role == "assistant":
                content: list[dict] = []

                if message.content:
                    content.append(
                        {
                            "text": message.content,
                        }
                    )

                for tool_call in message.metadata.get(
                    "tool_calls",
                    [],
                ):
                    content.append(
                        {
                            "toolUse": {
                                "toolUseId": tool_call.get(
                                    "tool_use_id",
                                    "",
                                ),
                                "name": tool_call.get("name", ""),
                                "input": tool_call.get("input", {}),
                            }
                        }
                    )

                serialized.append(
                    {
                        "role": "assistant",
                        "content": content,
                    }
                )

                index += 1
                continue

            serialized.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "text": message.content,
                        }
                    ],
                }
            )

            index += 1

        return serialized

    def serialize_tools(
        self,
        tools: list[Tool],
    ) -> list[dict]:
        return [
            {
                "toolSpec": {
                    "name": tool.name,
                    "description": tool.description,
                    "inputSchema": {
                        "json": tool.args_schema,
                    },
                }
            }
            for tool in tools
        ]

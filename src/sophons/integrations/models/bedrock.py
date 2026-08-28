from __future__ import annotations

from typing import Any

from sophons.integrations.models.adapters.bedrock import BedrockAdapter
from sophons.models.messages import Message
from sophons.tools.base import Tool


class BedrockModel:
    """Sophons chat-model implementation for Amazon Bedrock Converse."""

    def __init__(
        self,
        model: str,
        *,
        region: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        client: Any | None = None,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._adapter = BedrockAdapter()

        if client is not None:
            self._client = client
            return

        try:
            import boto3
        except ImportError as exc:
            raise ImportError(
                "BedrockModel requires the 'bedrock' extra. "
                "Install it with: uv sync --extra bedrock"
            ) from exc

        client_options: dict[str, Any] = {}

        if region is not None:
            client_options["region_name"] = region

        self._client = boto3.client(
            "bedrock-runtime",
            **client_options,
        )

    def invoke(
        self,
        messages: list[Message],
        tools: list[Tool] | None = None,
    ) -> Message:
        system_messages = [
            {
                "text": message.content,
            }
            for message in messages
            if message.role == "system" and message.content
        ]

        conversation_messages = [
            message for message in messages if message.role != "system"
        ]

        request: dict[str, Any] = {
            "modelId": self.model,
            "messages": self._adapter.serialize_messages(conversation_messages),
            "inferenceConfig": {
                "maxTokens": self.max_tokens,
                "temperature": self.temperature,
            },
        }

        if system_messages:
            request["system"] = system_messages

        if tools:
            request["toolConfig"] = {
                "tools": self._adapter.serialize_tools(tools),
            }

        response = self._client.converse(**request)

        return self._deserialize_response(response)

    def _deserialize_response(
        self,
        response: dict[str, Any],
    ) -> Message:
        output = response.get("output", {})
        message = output.get("message", {})
        content_blocks = message.get("content", [])

        text_parts = [block["text"] for block in content_blocks if "text" in block]

        tool_calls = [
            {
                "tool_use_id": block["toolUse"]["toolUseId"],
                "name": block["toolUse"]["name"],
                "input": block["toolUse"].get("input", {}),
            }
            for block in content_blocks
            if "toolUse" in block
        ]

        metadata: dict[str, Any] = {}

        if tool_calls:
            metadata["tool_calls"] = tool_calls

        usage = response.get("usage")

        if usage:
            metadata["usage"] = {
                "input_tokens": usage.get("inputTokens", 0),
                "output_tokens": usage.get("outputTokens", 0),
                "cache_read_tokens": usage.get(
                    "cacheReadInputTokens",
                    0,
                ),
                "cache_write_tokens": usage.get(
                    "cacheWriteInputTokens",
                    0,
                ),
            }

        stop_reason = response.get("stopReason")

        if stop_reason is not None:
            metadata["stop_reason"] = stop_reason

        metrics = response.get("metrics")

        if metrics is not None:
            metadata["provider_metrics"] = metrics

        return Message(
            role="assistant",
            content=" ".join(text_parts).strip(),
            metadata=metadata,
        )

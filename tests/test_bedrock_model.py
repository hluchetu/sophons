from __future__ import annotations

from typing import Any

from sophons.models import BedrockModel, Message


class FakeBedrockClient:
    def __init__(self, response: dict[str, Any]) -> None:
        self.response = response
        self.requests: list[dict[str, Any]] = []

    def converse(self, **request: Any) -> dict[str, Any]:
        self.requests.append(request)
        return self.response


def test_bedrock_model_serializes_request_and_text_response() -> None:
    client = FakeBedrockClient(
        {
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [{"text": "Python is required."}],
                }
            },
            "stopReason": "end_turn",
            "usage": {
                "inputTokens": 12,
                "outputTokens": 4,
                "cacheReadInputTokens": 2,
                "cacheWriteInputTokens": 1,
            },
            "metrics": {"latencyMs": 80},
        }
    )
    model = BedrockModel(
        model="test-model",
        client=client,
        max_tokens=256,
        temperature=0.2,
    )

    result = model.invoke(
        [
            Message(role="system", content="Extract job requirements."),
            Message(role="user", content="Analyze this Python role."),
        ]
    )

    assert client.requests == [
        {
            "modelId": "test-model",
            "messages": [
                {
                    "role": "user",
                    "content": [{"text": "Analyze this Python role."}],
                }
            ],
            "inferenceConfig": {
                "maxTokens": 256,
                "temperature": 0.2,
            },
            "system": [{"text": "Extract job requirements."}],
        }
    ]
    assert result == Message(
        role="assistant",
        content="Python is required.",
        metadata={
            "usage": {
                "input_tokens": 12,
                "output_tokens": 4,
                "cache_read_tokens": 2,
                "cache_write_tokens": 1,
            },
            "stop_reason": "end_turn",
            "provider_metrics": {"latencyMs": 80},
        },
    )


def test_bedrock_model_normalizes_tool_calls() -> None:
    client = FakeBedrockClient(
        {
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "toolUse": {
                                "toolUseId": "call-1",
                                "name": "search_candidates",
                                "input": {"skill": "Python"},
                            }
                        }
                    ],
                }
            },
            "stopReason": "tool_use",
        }
    )
    model = BedrockModel(model="test-model", client=client)

    result = model.invoke([Message(role="user", content="Find Python candidates.")])

    assert result.content == ""
    assert result.metadata == {
        "tool_calls": [
            {
                "tool_use_id": "call-1",
                "name": "search_candidates",
                "input": {"skill": "Python"},
            }
        ],
        "stop_reason": "tool_use",
    }

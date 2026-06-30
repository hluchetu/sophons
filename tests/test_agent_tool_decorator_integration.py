from __future__ import annotations

from sophons import Message
from sophons.agents import Agent
from sophons.tools import tool


class ToolCallingModel:
    def __init__(self) -> None:
        self.calls = 0
        self.messages_seen: list[list[Message]] = []

    def invoke(self, messages: list[Message]) -> Message:
        self.calls += 1
        self.messages_seen.append(messages)

        if self.calls == 1:
            return Message(
                role="assistant",
                content="",
                metadata={
                    "tool_calls": [
                        {
                            "tool_use_id": "call_1",
                            "name": "add",
                            "input": {"a": 2, "b": 3},
                        }
                    ]
                },
            )

        return Message(role="assistant", content="The answer is 5.")


def test_agent_executes_decorated_tool_from_model_tool_call() -> None:
    @tool
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    model = ToolCallingModel()
    agent = Agent(model=model, tools=[add])

    result = agent.run_sync("What is 2 + 3?")

    assert result.success is True
    assert result.message == "The answer is 5."
    assert result.metrics.model_calls == 2
    assert result.metrics.tool_calls == 1
    assert result.tool_uses[0].name == "add"
    assert result.tool_uses[0].input == {"a": 2, "b": 3}
    assert result.tool_results[0].status == "success"
    assert result.tool_results[0].content == '{"result": 5}'

    second_call_messages = model.messages_seen[1]
    assert second_call_messages[-1].role == "tool"
    assert second_call_messages[-1].content == '{"result": 5}'
    assert second_call_messages[-1].metadata == {
        "tool_use_id": "call_1",
        "status": "success",
    }

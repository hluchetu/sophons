"""Tool budgets apply to individual dispatches, not just model rounds."""
import pytest
from pydantic import BaseModel
from sophons.agents import Agent
from sophons.agents.responses import StopReason
from sophons.guardrails import ToolPermissionGuardrail
from sophons.agents.state import RunLimits
from sophons.models.messages import Message
from sophons.tools import tool


class BatchModel:
    def invoke(self, messages, tools=None):
        return Message(role="assistant", content="", metadata={"tool_calls": [
            {"tool_use_id": "first", "name": "echo", "input": {"value": "first"}},
            {"tool_use_id": "second", "name": "echo", "input": {"value": "second"}},
        ]})


def test_one_call_budget_stops_second_tool_in_the_same_batch():
    executed = []

    @tool
    def echo(value: str) -> str:
        """Record a synthetic execution and echo its value."""
        executed.append(value)
        return value

    result = Agent(model=BatchModel(), tools=[echo],
        limits=RunLimits(max_tool_calls=1)).run_sync("Synthetic question")

    assert executed == ["first"]  # the second function must never run
    assert result.metrics.tool_calls == 1
    assert result.success is False
    assert result.stop_reason.value == "max_tool_calls"
    assert [(item.tool_use_id, item.status) for item in result.tool_results] == [
        ("first", "success"), ("second", "error")
    ]
    assert "budget" in result.tool_results[1].content.lower()


class ScriptedModel:
    def __init__(self, batches):
        self.batches = list(batches)
        self.calls = 0

    def invoke(self, messages, tools=None):
        batch = self.batches[self.calls]
        self.calls += 1
        if isinstance(batch, str):
            return Message(role="assistant", content=batch)
        return Message(role="assistant", content="", metadata={"tool_calls": batch})


def requests(*values):
    return [{"tool_use_id": value, "name": "echo", "input": {"value": value}} for value in values]


def echo_tool(executed):
    @tool
    def echo(value: str) -> str:
        """Record a synthetic execution and return the value."""
        executed.append(value)
        return value
    return echo


def test_budget_accumulates_across_model_rounds():
    executed = []
    model = ScriptedModel([requests("one", "two"), requests("three", "four")])
    result = Agent(model=model, tools=[echo_tool(executed)],
        limits=RunLimits(max_tool_calls=3)).run_sync("Question")
    assert executed == ["one", "two", "three"]
    assert model.calls == 2 and result.metrics.tool_calls == 3
    assert result.stop_reason == StopReason.MAX_TOOL_CALLS
    assert result.tool_results[-1].tool_use_id == "four"
    assert result.tool_results[-1].status == "error"


def test_exact_batch_capacity_is_not_exceeded():
    executed = []
    result = Agent(model=BatchModel(), tools=[echo_tool(executed)],
        limits=RunLimits(max_tool_calls=2)).run_sync("Question")
    assert executed == ["first", "second"] and result.metrics.tool_calls == 2
    assert all(item.status == "success" for item in result.tool_results)
    assert result.stop_reason == StopReason.MAX_TOOL_CALLS


def test_run_finishes_normally_when_budget_remains_for_final_model_response():
    executed = []
    model = ScriptedModel([requests("one"), "Completed"])
    result = Agent(model=model, tools=[echo_tool(executed)],
        limits=RunLimits(max_tool_calls=2)).run_sync("Question")
    assert executed == ["one"]
    assert result.success and result.message == "Completed"
    assert result.metrics.tool_calls == 1


def test_zero_budget_stops_before_model_or_tool_work():
    executed = []
    model = ScriptedModel([requests("one")])
    result = Agent(model=model, tools=[echo_tool(executed)],
        limits=RunLimits(max_tool_calls=0)).run_sync("Question")
    assert model.calls == 0 and not executed
    assert result.metrics.tool_calls == 0
    assert result.stop_reason == StopReason.MAX_TOOL_CALLS


def test_budget_is_fresh_for_each_run():
    executed = []
    agent = Agent(model=BatchModel(), tools=[echo_tool(executed)], limits=RunLimits(max_tool_calls=1))
    first, second = agent.run_sync("First"), agent.run_sync("Second")
    assert executed == ["first", "first"]
    assert first.metrics.tool_calls == second.metrics.tool_calls == 1


def test_denied_dispatch_consumes_slot_but_skipped_request_does_not():
    executed = []
    result = Agent(model=BatchModel(), tools=[echo_tool(executed)],
        guardrails=[ToolPermissionGuardrail(denied={"echo"})],
        limits=RunLimits(max_tool_calls=1)).run_sync("Question")
    assert not executed and result.metrics.tool_calls == 1
    assert result.metrics.per_tool["echo"].calls == 1
    assert len(result.tool_results) == 2
    assert "guardrail" in result.tool_results[0].content.lower()
    assert "budget" in result.tool_results[1].content.lower()


@pytest.mark.asyncio
async def test_async_tools_obey_the_same_batch_budget():
    executed = []
    class AsyncEcho:
        name = "echo"
        description = "Synthetic async echo."
        args_schema = {"type": "object", "properties": {"value": {"type": "string"}}, "required": ["value"]}

        async def call(self, args):
            executed.append(args["value"])
            return {"result": args["value"]}

    result = await Agent(model=BatchModel(), tools=[AsyncEcho()], limits=RunLimits(max_tool_calls=1)).run("Question")
    assert executed == ["first"] and result.metrics.tool_calls == 1



class FinalAnswer(BaseModel):
    text: str


def test_structured_response_consumes_one_budget_slot():
    model = ScriptedModel([[{"tool_use_id": "final", "name": "structured_response", "input": {"text": "Done"}}]])
    result = Agent(model=model, output_type=FinalAnswer, limits=RunLimits(max_tool_calls=1)).run_sync("Question")
    assert result.success and result.output == FinalAnswer(text="Done")
    assert result.metrics.tool_calls == 1


def test_invalid_structured_response_cannot_retry_past_tool_budget():
    model = ScriptedModel([[{"tool_use_id": "invalid", "name": "structured_response", "input": {}}]])
    result = Agent(model=model, output_type=FinalAnswer, limits=RunLimits(max_tool_calls=1)).run_sync("Question")
    assert model.calls == 1 and result.metrics.tool_calls == 1
    assert not result.success and result.output is None
    assert result.stop_reason == StopReason.MAX_TOOL_CALLS


@pytest.mark.parametrize("limit", [-1, 1.5, True, "2"])
def test_invalid_tool_call_budget_is_rejected(limit):
    with pytest.raises(ValueError, match="nonnegative integer"):
        RunLimits(max_tool_calls=limit)

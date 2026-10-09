"""Regression tests: structured final answers must pass output policies too."""
import json

import pytest
from pydantic import BaseModel

from sophons.agents import Agent
from sophons.agents.responses import StopReason
from sophons.guardrails import GuardrailChain, GuardrailDecision
from sophons.models.messages import Message


class Answer(BaseModel):
    text: str


class StructuredModel:
    """Fake model: return a valid output-tool call without a provider request."""

    def invoke(self, messages, tools=None):
        return Message(role="assistant", content="", metadata={"tool_calls": [
            {"tool_use_id": "answer-1", "name": "structured_response",
             "input": {"text": "Synthetic answer"}}
        ]})


class BlockFinalOutput:
    name = "block-final-output"

    def __init__(self):
        self.checked_outputs = []

    async def check(self, value, *, context):
        if context.boundary == "output":
            self.checked_outputs.append(value)
            return GuardrailDecision.block("Test policy", message="Answer blocked.")
        return GuardrailDecision.allow()


def test_structured_answer_cannot_bypass_output_guardrail():
    guard = BlockFinalOutput()
    agent = Agent(model=StructuredModel(), output_type=Answer, guardrails=[guard])

    result = agent.run_sync("A synthetic question")

    # The policy must see the final JSON, just as text-output policies see text.
    assert len(guard.checked_outputs) == 1
    assert json.loads(guard.checked_outputs[0]) == {"text": "Synthetic answer"}
    assert result.success is False
    assert result.stop_reason == StopReason.GUARDRAIL
    assert result.output is None
    assert result.message == "Answer blocked."


class FinalPolicy:
    name = "final-policy"

    def __init__(self, decision):
        self.decision = decision

    async def check(self, value, *, context):
        return self.decision if context.boundary == "output" else GuardrailDecision.allow()


def test_allowed_structured_answer_preserves_typed_output():
    result = Agent(model=StructuredModel(), output_type=Answer,
        guardrails=[FinalPolicy(GuardrailDecision.allow())]).run_sync("Question")
    assert result.success is True
    assert result.output == Answer(text="Synthetic answer")
    assert json.loads(result.message) == result.output.model_dump()


@pytest.mark.parametrize("transformed", [
    '{"text": "Redacted answer"}',
    {"text": "Redacted answer"},
    Answer(text="Redacted answer"),
])
def test_transformed_structured_answer_is_revalidated(transformed):
    policy = FinalPolicy(GuardrailDecision.transform(transformed, reason="Test redaction"))
    result = Agent(model=StructuredModel(), output_type=Answer,
        guardrails=[policy]).run_sync("Question")
    assert result.success is True
    assert result.output == Answer(text="Redacted answer")
    assert json.loads(result.message) == {"text": "Redacted answer"}
    assert json.loads(result.tool_results[-1].content) == {"text": "Redacted answer"}


@pytest.mark.parametrize("transformed", ["not JSON", {}, {"text": 42}, None])
def test_invalid_transformation_cannot_escape_the_output_schema(transformed):
    policy = FinalPolicy(GuardrailDecision.transform(transformed, reason="Test transformation"))
    result = Agent(model=StructuredModel(), output_type=Answer,
        guardrails=[policy]).run_sync("Question")
    assert result.success is False
    assert result.stop_reason == StopReason.GUARDRAIL
    assert result.output is None
    assert "required output schema" in result.message
    assert not result.tool_results  # never report the rejected output as accepted


def test_confirmation_required_at_output_is_not_accepted():
    policy = FinalPolicy(GuardrailDecision.confirm("Review required"))
    result = Agent(model=StructuredModel(), output_type=Answer,
        guardrails=[policy]).run_sync("Question")
    assert result.success is False
    assert result.stop_reason == StopReason.GUARDRAIL
    assert result.output is None


def test_shadow_output_policy_is_observed_without_blocking():
    guard = BlockFinalOutput()
    chain = GuardrailChain([guard], mode="shadow")
    result = Agent(model=StructuredModel(), output_type=Answer,
        guardrails=chain).run_sync("Question")
    assert len(guard.checked_outputs) == 1
    assert result.success is True
    assert result.output == Answer(text="Synthetic answer")

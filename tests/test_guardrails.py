from __future__ import annotations

from typing import Any

import pytest

from sophons.agents import Agent
from sophons.agents.responses import StopReason
from sophons.guardrails import (
    CREDIT_CARD,
    Guardrail,
    GuardrailChain,
    GuardrailContext,
    GuardrailDecision,
    PatternGuardrail,
    ToolPermissionGuardrail,
)
from sophons.models import Message
from sophons.tools import tool

# ---------------------------------------------------------------------------
# Chain semantics
# ---------------------------------------------------------------------------


class Always:
    def __init__(self, name: str, decision: GuardrailDecision) -> None:
        self.name = name
        self._decision = decision
        self.saw: list[Any] = []

    async def check(self, value, *, context) -> GuardrailDecision:
        self.saw.append(value)
        return self._decision


INPUT_CTX = GuardrailContext(boundary="input")


@pytest.mark.asyncio
async def test_chain_stops_at_first_block() -> None:
    blocker = Always("b", GuardrailDecision.block("nope"))
    after = Always("after", GuardrailDecision.allow())
    chain = GuardrailChain([blocker, after])

    decision = await chain.check("value", context=INPUT_CTX)

    assert decision.action == "block"
    assert decision.metadata["guardrail"] == "b"
    assert after.saw == []  # never reached


@pytest.mark.asyncio
async def test_chain_threads_transformed_value() -> None:
    redact = Always("r", GuardrailDecision.transform("clean", reason="scrub"))
    downstream = Always("d", GuardrailDecision.allow())
    chain = GuardrailChain([redact, downstream])

    decision = await chain.check("dirty", context=INPUT_CTX)

    assert decision.action == "transform"
    assert decision.transformed == "clean"
    assert downstream.saw == ["clean"]  # next guardrail sees the new value


@pytest.mark.asyncio
async def test_shadow_mode_records_but_never_enforces() -> None:
    blocker = Always("b", GuardrailDecision.block("would block"))
    chain = GuardrailChain([blocker], mode="shadow")

    decision = await chain.check("value", context=INPUT_CTX)

    assert decision.action == "allow"
    assert decision.metadata["decisions"] == [
        {"guardrail": "b", "action": "block", "reason": "would block"}
    ]


# ---------------------------------------------------------------------------
# Loop integration
# ---------------------------------------------------------------------------


class ScriptedModel:
    """Requests a tool on the first call, answers on the second."""

    def __init__(self, tool_name: str = "refund_order", answer: str = "done") -> None:
        self.tool_name = tool_name
        self.answer = answer
        self.tool_results_seen: list[str] = []

    def invoke(self, messages, tools=None):
        last = messages[-1]
        if last.role == "tool":
            self.tool_results_seen.append(last.content)
        if not any(m.role == "tool" for m in messages):
            return Message(
                role="assistant",
                content="",
                metadata={
                    "tool_calls": [
                        {
                            "tool_use_id": "t1",
                            "name": self.tool_name,
                            "input": {"amount": 500},
                        }
                    ]
                },
            )
        return Message(role="assistant", content=self.answer)


@tool
def refund_order(amount: float) -> str:
    """Refund the given amount."""
    return f"refunded {amount}"


def _capped_agent(model) -> Agent:
    return Agent(
        model=model,
        tools=[refund_order],
        guardrails=[
            ToolPermissionGuardrail(
                argument_rules={
                    "refund_order": lambda a: (
                        "amount exceeds limit" if a.get("amount", 0) > 100 else None
                    )
                }
            )
        ],
    )


def test_input_block_ends_run_with_guardrail_stop_reason() -> None:
    agent = Agent(
        model=ScriptedModel(),
        guardrails=[
            PatternGuardrail(
                patterns={"card": CREDIT_CARD},
                action="block",
                boundaries=("input",),
                message="No card numbers please.",
            )
        ],
    )

    result = agent.run_sync("charge 4111 1111 1111 1111")

    assert result.stop_reason is StopReason.GUARDRAIL
    assert result.message == "No card numbers please."
    assert result.success is False
    assert result.metrics.model_calls == 0  # blocked before any model call


def test_blocked_tool_never_executes_and_model_sees_reason() -> None:
    model = ScriptedModel()
    agent = _capped_agent(model)

    result = agent.run_sync("refund 500 please")

    assert result.stop_reason is StopReason.END_TURN  # run continues
    assert result.tool_results[0].status == "error"
    assert "blocked by guardrail" in result.tool_results[0].content
    assert any("blocked by guardrail" in c for c in model.tool_results_seen)


def test_output_transform_rewrites_final_answer() -> None:
    agent = Agent(
        model=ScriptedModel(tool_name="refund_order", answer="card 4111 1111 1111 1111 refunded"),
        tools=[refund_order],
        guardrails=[
            PatternGuardrail(patterns={"card": CREDIT_CARD}, boundaries=("output",))
        ],
    )

    result = agent.run_sync("refund me")

    assert "[redacted]" in result.message
    assert "4111" not in result.message


def test_no_guardrails_means_no_behavior_change() -> None:
    agent = Agent(model=ScriptedModel(), tools=[refund_order])

    result = agent.run_sync("refund 500 please")

    assert result.stop_reason is StopReason.END_TURN
    assert result.tool_results[0].status == "success"


# ---------------------------------------------------------------------------
# ToolPermissionGuardrail policy matrix
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_denied_beats_allowed() -> None:
    guard = ToolPermissionGuardrail(allowed={"x"}, denied={"x"})

    decision = await guard.check(
        {}, context=GuardrailContext(boundary="tool", tool_name="x")
    )

    assert decision.action == "block"
    assert "denied" in decision.reason


@pytest.mark.asyncio
async def test_allowlist_blocks_unlisted_tools() -> None:
    guard = ToolPermissionGuardrail(allowed={"search"})

    decision = await guard.check(
        {}, context=GuardrailContext(boundary="tool", tool_name="send_email")
    )

    assert decision.action == "block"


@pytest.mark.asyncio
async def test_permission_guard_abstains_off_boundary() -> None:
    guard = ToolPermissionGuardrail(denied={"x"})

    decision = await guard.check("anything", context=INPUT_CTX)

    assert decision.action == "allow"


@pytest.mark.asyncio
async def test_protocol_conformance() -> None:
    assert isinstance(ToolPermissionGuardrail(), Guardrail)
    assert isinstance(PatternGuardrail(patterns={"c": CREDIT_CARD}), Guardrail)

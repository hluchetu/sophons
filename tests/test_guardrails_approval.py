from __future__ import annotations

import pytest

from sophons.agents import Agent
from sophons.guardrails import (
    ApprovalDecision,
    ApprovalGuardrail,
    ApprovalRequest,
    Approver,
    CallbackApprover,
    ConsoleApprover,
    GuardrailChain,
    GuardrailContext,
    GuardrailDecision,
)
from sophons.models import Message
from sophons.tools import tool

# ---------------------------------------------------------------------------
# Shapes and fail-safe semantics
# ---------------------------------------------------------------------------


def test_confirm_is_not_allowed_until_approved() -> None:
    decision = GuardrailDecision.confirm("needs a human")

    assert decision.action == "confirm"
    assert decision.allowed is False  # fail-safe by construction


def test_approvers_satisfy_protocol() -> None:
    async def cb(request: ApprovalRequest) -> ApprovalDecision:
        return ApprovalDecision(approved=True, approver="t")

    assert isinstance(CallbackApprover(cb), Approver)
    assert isinstance(ConsoleApprover(), Approver)


@pytest.mark.asyncio
async def test_chain_propagates_confirm_and_stops() -> None:
    class Confirming:
        name = "c"

        async def check(self, value, *, context):
            return GuardrailDecision.confirm("ask a human")

    class After:
        name = "after"
        called = False

        async def check(self, value, *, context):
            After.called = True
            return GuardrailDecision.allow()

    chain = GuardrailChain([Confirming(), After()])
    decision = await chain.check({}, context=GuardrailContext(boundary="tool"))

    assert decision.action == "confirm"
    assert After.called is False


# ---------------------------------------------------------------------------
# ApprovalGuardrail predicate
# ---------------------------------------------------------------------------

RULES = {
    "refund_order": lambda args: (
        f"amount {args.get('amount')} exceeds the auto-approve limit"
        if args.get("amount", 0) > 100
        else None
    )
}


@pytest.mark.asyncio
async def test_rule_over_limit_confirms_under_limit_allows() -> None:
    guard = ApprovalGuardrail(rules=RULES)
    ctx = GuardrailContext(boundary="tool", tool_name="refund_order")

    over = await guard.check({"amount": 250.0}, context=ctx)
    under = await guard.check({"amount": 30.0}, context=ctx)

    assert over.action == "confirm"
    assert "250.0" in over.reason
    assert under.action == "allow"


@pytest.mark.asyncio
async def test_unruled_tools_and_other_boundaries_allow() -> None:
    guard = ApprovalGuardrail(rules=RULES)

    other_tool = await guard.check(
        {}, context=GuardrailContext(boundary="tool", tool_name="search_docs")
    )
    other_boundary = await guard.check(
        "text", context=GuardrailContext(boundary="input")
    )

    assert other_tool.action == "allow"
    assert other_boundary.action == "allow"


# ---------------------------------------------------------------------------
# Loop integration: approve / deny / fail-safe
# ---------------------------------------------------------------------------


class ScriptedModel:
    def invoke(self, messages, tools=None):
        if not any(m.role == "tool" for m in messages):
            return Message(
                role="assistant",
                content="",
                metadata={
                    "tool_calls": [
                        {
                            "tool_use_id": "t1",
                            "name": "refund_order",
                            "input": {"amount": 250.0},
                        }
                    ]
                },
            )
        return Message(role="assistant", content="done")


def _agent(approver) -> Agent:
    @tool
    def refund_order(amount: float) -> str:
        """Refund an amount."""
        return f"refunded {amount}"

    return Agent(
        model=ScriptedModel(),
        tools=[refund_order],
        guardrails=[ApprovalGuardrail(rules=RULES)],
        approver=approver,
    )


def _approver(approved: bool, note: str = "") -> CallbackApprover:
    async def cb(request: ApprovalRequest) -> ApprovalDecision:
        return ApprovalDecision(approved=approved, approver="mgr-7", note=note)

    return CallbackApprover(cb, name="mgr")


def test_approved_call_executes() -> None:
    result = _agent(_approver(True)).run_sync("refund 250")

    assert result.tool_results[0].status == "success"
    assert "refunded 250" in result.tool_results[0].content


def test_denied_call_carries_the_humans_note() -> None:
    result = _agent(_approver(False, "over my limit too")).run_sync("refund 250")

    assert result.tool_results[0].status == "error"
    assert "denied by mgr-7" in result.tool_results[0].content
    assert "over my limit too" in result.tool_results[0].content


def test_no_approver_fails_safe() -> None:
    result = _agent(None).run_sync("refund 250")

    assert result.tool_results[0].status == "error"
    assert "no approver is configured" in result.tool_results[0].content

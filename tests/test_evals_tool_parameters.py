from __future__ import annotations

import pytest

from sophons.evals import ToolParameterEvaluator


@pytest.mark.asyncio
async def test_right_tool_wrong_argument_fails_with_named_field() -> None:
    evaluator = ToolParameterEvaluator()

    result = await evaluator.evaluate(
        "q",
        "a",
        expected_tool_calls=[
            {"name": "refund_order", "args": {"order_id": "ord_123", "amount": 49.99}}
        ],
        actual_tool_calls=[
            {"name": "refund_order", "args": {"order_id": "ord_999", "amount": 49.99}}
        ],
    )

    score = result.scores[0]
    assert score.passed is False
    assert score.metadata["argument_mismatches"] == [
        {
            "tool": "refund_order",
            "field": "order_id",
            "expected": "ord_123",
            "actual": "ord_999",
        }
    ]
    assert "refund_order.order_id" in score.reason


@pytest.mark.asyncio
async def test_matching_call_passes_and_ignores_extra_actual_fields() -> None:
    evaluator = ToolParameterEvaluator()

    result = await evaluator.evaluate(
        "q",
        "a",
        expected_tool_calls=[{"name": "search_docs", "args": {"query": "refunds"}}],
        actual_tool_calls=[
            {"name": "search_docs", "args": {"query": "refunds", "limit": 4}}
        ],
    )

    assert result.scores[0].passed is True
    assert result.scores[0].score == 1.0


@pytest.mark.asyncio
async def test_missing_call_and_missing_field_are_reported() -> None:
    evaluator = ToolParameterEvaluator()

    result = await evaluator.evaluate(
        "q",
        "a",
        expected_tool_calls=[
            {"name": "search_docs", "args": {"query": "refunds"}},
            {"name": "create_ticket", "args": {"priority": "high"}},
        ],
        actual_tool_calls=[{"name": "search_docs", "args": {}}],
    )

    score = result.scores[0]
    assert score.passed is False
    assert score.metadata["missing_calls"] == ["create_ticket"]
    assert score.metadata["argument_mismatches"][0]["actual"] == "<missing>"
    assert score.score == 0.0  # neither expected call fully matched


@pytest.mark.asyncio
async def test_missing_expectations_raise() -> None:
    evaluator = ToolParameterEvaluator()

    with pytest.raises(ValueError, match="expected_tool_calls"):
        await evaluator.evaluate("q", "a", actual_tool_calls=[])

from __future__ import annotations

from typing import Any

from sophons.evals.base import EvalResult, EvalScore

EVALUATOR_VERSION = "v0"


class ToolParameterEvaluator:
    """
    Deterministic check that tools were called with the right arguments.

    Trajectory answers "were the right tools called?"; this answers the
    failure mode trajectory cannot see: the right tool with the wrong
    arguments — ``refund_order(order_id="ord_999")`` when the case expected
    ``ord_123``. Same tool list, catastrophically different outcome.

    Expected and actual calls are dicts of ``{"name": str, "args": dict}``.
    Each expected call is matched to the first unmatched actual call with
    the same name, then compared field by field: every expected field must
    be present and equal. Actual fields the case does not mention are
    ignored — the test pins what matters, not the whole payload.
    """

    async def evaluate(
        self,
        question: str,
        answer: str,
        *,
        context: str | None = None,
        reference: str | None = None,
        tool_calls: list[str] | None = None,
        expected_tools: list[str] | None = None,
        expected_tool_calls: list[dict[str, Any]] | None = None,
        actual_tool_calls: list[dict[str, Any]] | None = None,
    ) -> EvalResult:
        if expected_tool_calls is None:
            raise ValueError(
                "ToolParameterEvaluator requires expected_tool_calls on the "
                "test case."
            )
        actual = list(actual_tool_calls or [])

        mismatches: list[dict[str, Any]] = []
        missing_calls: list[str] = []
        matched = 0

        unclaimed = list(actual)
        for expected in expected_tool_calls:
            name = expected["name"]
            candidate = next(
                (call for call in unclaimed if call.get("name") == name), None
            )
            if candidate is None:
                missing_calls.append(name)
                continue
            unclaimed.remove(candidate)

            call_mismatches = _compare_args(
                name, expected.get("args", {}), candidate.get("args", {})
            )
            if call_mismatches:
                mismatches.extend(call_mismatches)
            else:
                matched += 1

        passed = not mismatches and not missing_calls
        total = len(expected_tool_calls)
        score = EvalScore(
            dimension="tool_parameters",
            passed=passed,
            score=matched / total if total else 1.0,
            reason=_summarize(missing_calls, mismatches),
            metadata={
                "evaluator": "ToolParameterEvaluator",
                "evaluator_version": EVALUATOR_VERSION,
                "expected_tool_calls": expected_tool_calls,
                "actual_tool_calls": actual,
                "missing_calls": missing_calls,
                "argument_mismatches": mismatches,
            },
        )
        return EvalResult(question=question, answer=answer, scores=[score])


def _compare_args(
    tool: str, expected: dict[str, Any], actual: dict[str, Any]
) -> list[dict[str, Any]]:
    """Every expected field must be present and equal in the actual args."""
    mismatches = []
    for fieldname, expected_value in expected.items():
        actual_value = actual.get(fieldname, "<missing>")
        if actual_value != expected_value:
            mismatches.append(
                {
                    "tool": tool,
                    "field": fieldname,
                    "expected": expected_value,
                    "actual": actual_value,
                }
            )
    return mismatches


def _summarize(missing: list[str], mismatches: list[dict[str, Any]]) -> str:
    if not missing and not mismatches:
        return "every expected call was made with the expected arguments"
    parts = []
    if missing:
        parts.append(f"missing calls: {', '.join(missing)}")
    for m in mismatches:
        parts.append(
            f"{m['tool']}.{m['field']} expected {m['expected']!r} "
            f"got {m['actual']!r}"
        )
    return "; ".join(parts)

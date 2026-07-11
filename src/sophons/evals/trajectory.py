from __future__ import annotations

from collections import Counter
from typing import Literal

from sophons.evals.base import EvalResult, EvalScore

TrajectoryMode = Literal["exact", "in-order", "any-order"]

EVALUATOR_VERSION = "v0"


class TrajectoryEvaluator:
    """
    Deterministic check of the tool-call path — no judge involved.

    Compares the tools the agent actually called against the tools the
    test case expected, under one of three strictness modes:

    - ``exact``:     same tools, same order, nothing extra.
    - ``in-order``:  expected tools appear in order; extra calls allowed.
    - ``any-order``: expected tools all appear; order and extras ignored.
    """

    def __init__(self, mode: TrajectoryMode = "exact") -> None:
        self.mode = mode

    async def evaluate(
        self,
        question: str,
        answer: str,
        *,
        context: str | None = None,
        reference: str | None = None,
        tool_calls: list[str] | None = None,
        expected_tools: list[str] | None = None,
    ) -> EvalResult:
        if expected_tools is None:
            raise ValueError(
                "TrajectoryEvaluator requires expected_tools on the test case."
            )
        actual = tool_calls or []

        if self.mode == "exact":
            passed = actual == expected_tools
        elif self.mode == "in-order":
            passed = _is_subsequence(expected_tools, actual)
        else:
            passed = not Counter(expected_tools) - Counter(actual)

        found = Counter(expected_tools) & Counter(actual)
        coverage = sum(found.values()) / len(expected_tools) if expected_tools else 1.0
        missing = list((Counter(expected_tools) - Counter(actual)).elements())
        extra = list((Counter(actual) - Counter(expected_tools)).elements())

        score = EvalScore(
            dimension="trajectory",
            passed=passed,
            score=1.0 if passed else coverage,
            reason=(f"mode={self.mode} expected={expected_tools} actual={actual}"),
            metadata={
                "evaluator": "TrajectoryEvaluator",
                "evaluator_version": EVALUATOR_VERSION,
                "mode": self.mode,
                "expected_tools": expected_tools,
                "actual_tools": actual,
                "missing_tools": missing,
                "extra_tools": extra,
                "coverage": coverage,
            },
        )
        return EvalResult(question=question, answer=answer, scores=[score])


def _is_subsequence(expected: list[str], actual: list[str]) -> bool:
    """True if ``expected`` appears within ``actual`` in order, gaps allowed."""
    it = iter(actual)
    return all(tool in it for tool in expected)

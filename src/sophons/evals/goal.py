from __future__ import annotations

from typing import Any

from sophons.evals.base import EvalResult
from sophons.evals.judges import judge_dimension

_GOAL_STEPS = """\
1. State what a fully successful response to the question would accomplish.
2. Judge whether the answer accomplishes it end to end — a partial or \
hedged answer does not count.
3. If a reference answer is provided, use it as the bar for "fully".
4. passed = true only for complete success. score = 1.0 if passed else 0.0."""

EVALUATOR_VERSION = "v0"


class GoalEvaluator:
    """
    Did the task fully succeed, end to end? Deliberately blunt: pass/fail,
    no partial credit. Its power is aggregate — goal success *rate* across
    a test set, tracked over time.
    """

    def __init__(self, model: Any) -> None:
        self.model = model

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
        materials = {"question": question, "answer": answer}
        if reference is not None:
            materials["reference"] = reference

        score = await judge_dimension(
            self.model,
            dimension="goal",
            steps=_GOAL_STEPS,
            materials=materials,
            metadata={
                "evaluator": "GoalEvaluator",
                "evaluator_version": EVALUATOR_VERSION,
                "reference_provided": reference is not None,
            },
        )
        return EvalResult(question=question, answer=answer, scores=[score])

from __future__ import annotations

from typing import Any

from sophons.evals.base import EvalResult
from sophons.evals.judges import judge_dimension


_ANSWER_CORRECTNESS_STEPS = """\
1. List the factual claims made in the reference answer.
2. For each claim, check whether the answer asserts the same thing, \
in any wording.
3. Check whether the answer asserts anything the reference contradicts.
4. passed = true only if every reference claim is present and nothing \
is contradicted. score = fraction of reference claims present."""

EVALUATOR_VERSION = "v0"


class AnswerCorrectnessEvaluator:
    """
    Does the final answer match the reference answer?

    This is a post-generation evaluator for datasets with gold answers.
    It judges factual correctness against a reference, not whether the
    answer is grounded in retrieved context. Use FaithfulnessEvaluator for
    grounding.
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
        if reference is None:
            raise ValueError("AnswerCorrectnessEvaluator requires a reference answer.")

        score = await judge_dimension(
            self.model,
            dimension="answer_correctness",
            steps=_ANSWER_CORRECTNESS_STEPS,
            materials={
                "question": question,
                "answer": answer,
                "reference": reference,
            },
            metadata={
                "evaluator": "AnswerCorrectnessEvaluator",
                "evaluator_version": EVALUATOR_VERSION,
                "input_fields": ["question", "answer", "reference"],
            },
        )
        return EvalResult(question=question, answer=answer, scores=[score])

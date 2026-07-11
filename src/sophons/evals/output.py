from __future__ import annotations

from typing import Any

from sophons.evals.base import EvalResult
from sophons.evals.judges import judge_dimension

_CORRECTNESS_STEPS = """\
1. List the factual claims made in the reference answer.
2. For each claim, check whether the answer asserts the same thing, \
in any wording.
3. Check whether the answer asserts anything the reference contradicts.
4. passed = true only if every reference claim is present and nothing \
is contradicted. score = fraction of reference claims present."""

_RELEVANCY_STEPS = """\
1. Identify what the question is actually asking for.
2. Check whether the answer addresses that, rather than a related topic, \
a refusal, or filler.
3. passed = true only if a person asking the question would consider it \
answered. score = how directly the answer addresses the question."""

EVALUATOR_VERSION = "v0"


class OutputEvaluator:
    """
    Judges the final answer: correctness against a reference, and
    relevancy to the question. Two dimensions, two separate judge calls.
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
    ) -> EvalResult:
        scores = [
            await judge_dimension(
                self.model,
                dimension="relevancy",
                steps=_RELEVANCY_STEPS,
                materials={"question": question, "answer": answer},
                metadata={
                    "evaluator": "OutputEvaluator",
                    "evaluator_version": EVALUATOR_VERSION,
                    "input_fields": ["question", "answer"],
                    "reference_provided": reference is not None,
                },
            )
        ]
        if reference is not None:
            scores.append(
                await judge_dimension(
                    self.model,
                    dimension="correctness",
                    steps=_CORRECTNESS_STEPS,
                    materials={
                        "question": question,
                        "answer": answer,
                        "reference": reference,
                    },
                    metadata={
                        "evaluator": "OutputEvaluator",
                        "evaluator_version": EVALUATOR_VERSION,
                        "input_fields": ["question", "answer", "reference"],
                        "reference_provided": True,
                    },
                )
            )
        return EvalResult(question=question, answer=answer, scores=scores)

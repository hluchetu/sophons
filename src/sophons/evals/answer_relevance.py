from __future__ import annotations

from typing import Any

from sophons.evals.base import EvalResult
from sophons.evals.judges import judge_dimension


_ANSWER_RELEVANCE_STEPS = """\
1. Read the user question.
2. Read the assistant answer.
3. Decide whether the answer directly addresses what the user asked.
4. Pass if the answer is on-topic, useful, and responds to the user's intent.
5. Fail if the answer dodges the question, answers a different question, \
is mostly generic, or omits the main request.
6. score = 1.0 for a directly useful answer, 0.5 for a partially useful \
answer, 0.0 for an irrelevant answer."""

EVALUATOR_VERSION = "v0"


class AnswerRelevanceEvaluator:
    """
    Does the final answer respond to the user's question?

    This is a post-generation RAG evaluator. It judges answer usefulness
    relative to the question, not whether the answer is grounded in context.
    Use FaithfulnessEvaluator for grounding.
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
        score = await judge_dimension(
            self.model,
            dimension="answer_relevance",
            steps=_ANSWER_RELEVANCE_STEPS,
            materials={
                "question": question,
                "answer": answer,
            },
            metadata={
                "evaluator": "AnswerRelevanceEvaluator",
                "evaluator_version": EVALUATOR_VERSION,
                "input_fields": ["question", "answer"],
            },
        )
        return EvalResult(question=question, answer=answer, scores=[score])

from __future__ import annotations

from typing import Any

from sophons.evals.base import EvalResult
from sophons.evals.judges import judge_dimension

_CONTEXT_RELEVANCE_STEPS = """\
1. Read the question.
2. Read the retrieved context.
3. Decide whether the context contains the facts needed to answer the question.
4. Pass only if the context directly supports an answer to the question.
5. Fail if the context is unrelated, generic, empty, or missing the key evidence.
6. score = 1.0 for directly useful context, 0.5 for partially useful context, 0.0 for irrelevant context."""

EVALUATOR_VERSION = "v0"


class ContextRelevanceEvaluator:
    """
    Is the retrieved context useful evidence for answering the question?

    This is a pre-generation RAG evaluator. It judges retrieval quality,
    not answer quality. Use it before generation in corrective RAG loops.
    """

    def __init__(self, model: Any) -> None:
        self.model = model

    async def evaluate(
        self,
        question: str,
        answer: str = "",
        *,
        context: str | None = None,
        reference: str | None = None,
        tool_calls: list[str] | None = None,
        expected_tools: list[str] | None = None,
        expected_tool_calls: list[dict[str, Any]] | None = None,
        actual_tool_calls: list[dict[str, Any]] | None = None,
    ) -> EvalResult:
        if context is None:
            raise ValueError("ContextRelevanceEvaluator requires retrieved context.")

        score = await judge_dimension(
            self.model,
            dimension="context_relevance",
            steps=_CONTEXT_RELEVANCE_STEPS,
            materials={
                "question": question,
                "context": context,
            },
            metadata={
                "evaluator": "ContextRelevanceEvaluator",
                "evaluator_version": EVALUATOR_VERSION,
                "input_fields": ["question", "context"],
            },
        )
        return EvalResult(question=question, answer=answer, scores=[score])

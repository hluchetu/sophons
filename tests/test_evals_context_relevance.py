from __future__ import annotations

from dataclasses import dataclass

import pytest

from sophons.evals import ContextRelevanceEvaluator
from sophons.models import Message


@dataclass
class FakeResponse:
    content: str


class FakeJudgeModel:
    def invoke(self, messages: list[Message]) -> FakeResponse:
        prompt = messages[-1].content.lower()
        if "allows returns within 14 days" in prompt:
            return FakeResponse(
                '{"reason": "The context contains the needed policy.", '
                '"passed": true, "score": 1.0}'
            )
        return FakeResponse(
            '{"reason": "The context does not answer the question.", '
            '"passed": false, "score": 0.0}'
        )


@pytest.mark.asyncio
async def test_context_relevance_passes_when_context_answers_question() -> None:
    evaluator = ContextRelevanceEvaluator(FakeJudgeModel())

    result = await evaluator.evaluate(
        "What is the refund policy?",
        context="The refund policy allows returns within 14 days.",
    )

    score = result.scores[0]
    assert result.passed is True
    assert score.dimension == "context_relevance"
    assert score.passed is True
    assert score.score == 1.0
    assert score.reason == "The context contains the needed policy."
    assert score.metadata["evaluator"] == "ContextRelevanceEvaluator"
    assert score.metadata["input_fields"] == ["question", "context"]


@pytest.mark.asyncio
async def test_context_relevance_fails_when_context_is_unrelated() -> None:
    evaluator = ContextRelevanceEvaluator(FakeJudgeModel())

    result = await evaluator.evaluate(
        "What is the refund policy?",
        context="Support tickets are answered within two business days.",
    )

    score = result.scores[0]
    assert result.passed is False
    assert score.dimension == "context_relevance"
    assert score.passed is False
    assert score.score == 0.0


@pytest.mark.asyncio
async def test_context_relevance_requires_context() -> None:
    evaluator = ContextRelevanceEvaluator(FakeJudgeModel())

    with pytest.raises(ValueError, match="requires retrieved context"):
        await evaluator.evaluate("What is the refund policy?")

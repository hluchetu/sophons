from __future__ import annotations

from dataclasses import dataclass

import pytest

from sophons.evals import AnswerRelevanceEvaluator
from sophons.models import Message


@dataclass
class FakeResponse:
    content: str


class FakeJudgeModel:
    def invoke(self, messages: list[Message]) -> FakeResponse:
        prompt = messages[-1].content.lower()
        if "return worn shoes" in prompt and "shoes can be returned" in prompt:
            return FakeResponse(
                '{"reason": "The answer directly addresses the return question.", '
                '"passed": true, "score": 1.0}'
            )
        return FakeResponse(
            '{"reason": "The answer discusses a different topic.", '
            '"passed": false, "score": 0.0}'
        )


@pytest.mark.asyncio
async def test_answer_relevance_passes_when_answer_addresses_question() -> None:
    evaluator = AnswerRelevanceEvaluator(FakeJudgeModel())

    result = await evaluator.evaluate(
        "Can I return worn shoes?",
        "Shoes can be returned only if the wear is from trying them on indoors.",
    )

    score = result.scores[0]
    assert result.passed is True
    assert score.dimension == "answer_relevance"
    assert score.passed is True
    assert score.score == 1.0
    assert score.reason == "The answer directly addresses the return question."
    assert score.metadata["evaluator"] == "AnswerRelevanceEvaluator"
    assert score.metadata["input_fields"] == ["question", "answer"]


@pytest.mark.asyncio
async def test_answer_relevance_fails_when_answer_is_off_topic() -> None:
    evaluator = AnswerRelevanceEvaluator(FakeJudgeModel())

    result = await evaluator.evaluate(
        "Can I return worn shoes?",
        "Standard shipping takes three to five business days.",
    )

    score = result.scores[0]
    assert result.passed is False
    assert score.dimension == "answer_relevance"
    assert score.passed is False
    assert score.score == 0.0


@pytest.mark.asyncio
async def test_answer_relevance_ignores_unused_inputs() -> None:
    evaluator = AnswerRelevanceEvaluator(FakeJudgeModel())

    result = await evaluator.evaluate(
        "Can I return worn shoes?",
        "Shoes can be returned only if the wear is from trying them on indoors.",
        context="Shipping policy.",
        reference="Shoes must be unworn unless defective.",
        tool_calls=["search_docs"],
    )

    score = result.scores[0]
    assert result.passed is True
    assert score.metadata["input_fields"] == ["question", "answer"]

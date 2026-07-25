from __future__ import annotations

from dataclasses import dataclass

import pytest

from sophons.evals import AnswerCorrectnessEvaluator
from sophons.models import Message


@dataclass
class FakeResponse:
    content: str


class FakeJudgeModel:
    def invoke(self, messages: list[Message]) -> FakeResponse:
        prompt = messages[-1].content.lower()
        answer = prompt.split("## answer", 1)[1].split("## reference", 1)[0]
        if "must be unworn" in answer:
            return FakeResponse(
                '{"reason": "The answer matches the reference policy.", '
                '"passed": true, "score": 1.0}'
            )
        return FakeResponse(
            '{"reason": "The answer contradicts the reference policy.", '
            '"passed": false, "score": 0.0}'
        )


@pytest.mark.asyncio
async def test_answer_correctness_passes_when_answer_matches_reference() -> None:
    evaluator = AnswerCorrectnessEvaluator(FakeJudgeModel())

    result = await evaluator.evaluate(
        "Can I return worn shoes?",
        "Shoes must be unworn unless they are defective.",
        reference="Shoes must be unworn unless they are defective.",
    )

    score = result.scores[0]
    assert result.passed is True
    assert score.dimension == "answer_correctness"
    assert score.passed is True
    assert score.score == 1.0
    assert score.reason == "The answer matches the reference policy."
    assert score.metadata["evaluator"] == "AnswerCorrectnessEvaluator"
    assert score.metadata["input_fields"] == ["question", "answer", "reference"]


@pytest.mark.asyncio
async def test_answer_correctness_fails_when_answer_contradicts_reference() -> None:
    evaluator = AnswerCorrectnessEvaluator(FakeJudgeModel())

    result = await evaluator.evaluate(
        "Can I return worn shoes?",
        "Shoes can be returned after outdoor wear.",
        reference="Shoes must be unworn unless they are defective.",
    )

    score = result.scores[0]
    assert result.passed is False
    assert score.dimension == "answer_correctness"
    assert score.passed is False
    assert score.score == 0.0


@pytest.mark.asyncio
async def test_answer_correctness_requires_reference() -> None:
    evaluator = AnswerCorrectnessEvaluator(FakeJudgeModel())

    with pytest.raises(ValueError, match="requires a reference answer"):
        await evaluator.evaluate(
            "Can I return worn shoes?",
            "Shoes must be unworn unless they are defective.",
        )

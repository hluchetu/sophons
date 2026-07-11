from __future__ import annotations

import asyncio
import json
from typing import Any

from sophons.evals.base import EvalResult, EvalScore
from sophons.evals.judges import judge_dimension
from sophons.models import Message

_DECOMPOSE_SYSTEM = (
    "You extract factual claims. Given an answer, list every atomic factual "
    "claim it asserts — one self-contained statement each, no opinions, no "
    "hedges. Do not include statements about what a document, context, or "
    "source does or does not mention — only claims about the world itself. "
    "Respond with only a JSON array of strings, no prose, no code fences."
)

_VERIFY_STEPS = """\
1. For each numbered claim, search the context for a passage that supports it.
2. A claim is supported only if the context states it or directly implies it \
— background knowledge does not count.
3. passed = true only if every claim is supported. \
score = supported claims / total claims. \
reason = list each unsupported claim."""


class FaithfulnessEvaluator:
    """
    Is the answer grounded in the retrieved context?

    Decompose-then-verify (the RAGAS pattern): split the answer into atomic
    claims, then check every claim against the context. Catches the mixed
    answer — mostly grounded, one invented sentence — that a single
    "is this faithful?" call waves through.
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
        if context is None:
            raise ValueError(
                "FaithfulnessEvaluator requires the retrieved context."
            )

        claims = await self._decompose(answer)
        if not claims:
            score = EvalScore(
                dimension="faithfulness",
                passed=True,
                score=1.0,
                reason="Answer asserts no factual claims.",
            )
            return EvalResult(question=question, answer=answer, scores=[score])

        numbered = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(claims))
        score = await judge_dimension(
            self.model,
            dimension="faithfulness",
            steps=_VERIFY_STEPS,
            materials={"claims": numbered, "context": context},
        )
        return EvalResult(question=question, answer=answer, scores=[score])

    async def _decompose(self, answer: str) -> list[str]:
        """Split the answer into atomic factual claims."""
        messages = [
            Message(role="system", content=_DECOMPOSE_SYSTEM),
            Message(role="user", content=answer),
        ]
        response = self.model.invoke(messages)
        if asyncio.iscoroutine(response):
            response = await response

        text = response.content.strip()
        if text.startswith("```"):
            text = text.strip("`").removeprefix("json").strip()
        try:
            claims = json.loads(text)
        except ValueError:
            return []
        return [str(c) for c in claims] if isinstance(claims, list) else []

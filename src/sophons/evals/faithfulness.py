from __future__ import annotations

import asyncio
import json
from typing import Any

from sophons.evals.base import EvalResult, EvalScore
from sophons.evals.judges import JudgeError
from sophons.models import Message


_DECOMPOSE_SYSTEM = (
    "You extract factual claims. Given an answer, list every atomic factual "
    "claim it asserts — one self-contained statement each, no opinions, no "
    "hedges. Do not include statements about what a document, context, or "
    "source does or does not mention — only claims about the world itself. "
    "Respond with only a JSON array of strings, no prose, no code fences."
)

_VERIFY_JSON_SYSTEM = (
    "You are a strict faithfulness evaluator. Given numbered claims and a "
    "context, verify each claim against the context. A claim is supported "
    "only if the context states it or directly implies it — background "
    "knowledge does not count. Respond with only a JSON object, no prose, "
    "no code fences, in this shape: "
    '{"reason": "<one-sentence summary>", "claims": [{"claim": "<claim>", '
    '"supported": true or false, "evidence": "<supporting quote from the '
    'context, or null>", "reason": "<why>"}]}'
)

EVALUATOR_VERSION = "v1"


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
        expected_tool_calls: list[dict[str, Any]] | None = None,
        actual_tool_calls: list[dict[str, Any]] | None = None,
    ) -> EvalResult:
        if context is None:
            raise ValueError("FaithfulnessEvaluator requires the retrieved context.")

        claims = await self._decompose(answer)
        if not claims:
            score = EvalScore(
                dimension="faithfulness",
                passed=True,
                score=1.0,
                reason="Answer asserts no factual claims.",
                metadata={
                    "evaluator": "FaithfulnessEvaluator",
                    "evaluator_version": EVALUATOR_VERSION,
                    "claims": [],
                },
            )
            return EvalResult(question=question, answer=answer, scores=[score])

        score = await self._verify_claims(claims, context)
        return EvalResult(question=question, answer=answer, scores=[score])

    async def _verify_claims(self, claims: list[str], context: str) -> EvalScore:
        """Verify each claim against the context; one verdict per claim."""
        numbered = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(claims))
        messages = [
            Message(role="system", content=_VERIFY_JSON_SYSTEM),
            Message(
                role="user",
                content=f"# Claims\n{numbered}\n\n# Context\n{context}",
            ),
        ]

        last_error: Exception | None = None
        for _ in range(2):  # one retry on a malformed reply
            response = self.model.invoke(messages)
            if asyncio.iscoroutine(response):
                response = await response
            try:
                summary, verdicts = _parse_claim_verdicts(
                    response.content, expected_count=len(claims)
                )
            except ValueError as error:
                last_error = error
                continue

            supported = sum(1 for v in verdicts if v["supported"])
            return EvalScore(
                dimension="faithfulness",
                passed=supported == len(verdicts),
                score=supported / len(verdicts),
                reason=summary,
                metadata={
                    "evaluator": "FaithfulnessEvaluator",
                    "evaluator_version": EVALUATOR_VERSION,
                    "claims": claims,
                    "claim_verdicts": verdicts,
                },
            )

        raise JudgeError(
            f"faithfulness verifier returned no usable verdicts: {last_error}"
        )

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


def _parse_claim_verdicts(
    content: str, *, expected_count: int
) -> tuple[str, list[dict[str, Any]]]:
    """Parse the verifier's JSON reply. Raises ValueError on any bad shape."""
    text = content.strip()
    if text.startswith("```"):
        text = text.strip("`").removeprefix("json").strip()

    data = json.loads(text)  # ValueError on garbage
    if not isinstance(data, dict) or not isinstance(data.get("claims"), list):
        raise ValueError("expected a JSON object with a 'claims' array")

    raw = data["claims"]
    if len(raw) != expected_count:
        raise ValueError(
            f"expected {expected_count} claim verdicts, got {len(raw)}"
        )

    verdicts: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict) or not isinstance(item.get("supported"), bool):
            raise ValueError("each claim verdict needs a boolean 'supported'")
        evidence = item.get("evidence")
        verdicts.append(
            {
                "claim": str(item.get("claim", "")),
                "supported": item["supported"],
                "evidence": str(evidence) if evidence is not None else None,
                "reason": str(item.get("reason", "")).strip(),
            }
        )

    return str(data.get("reason", "")).strip(), verdicts

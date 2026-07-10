from __future__ import annotations

import asyncio
import json
from typing import Any

from sophons.evals.base import EvalScore
from sophons.models import Message

_JUDGE_SYSTEM = (
    "You are a strict evaluator. Follow the grading steps exactly. "
    "Respond with only a JSON object, no prose, no code fences: "
    '{"passed": true or false, "score": <number 0.0-1.0>, "reason": "<one sentence>"}'
)


class JudgeError(Exception):
    """The judge model did not return a usable verdict."""


async def judge_dimension(
    model: Any,
    *,
    dimension: str,
    steps: str,
    materials: dict[str, str],
) -> EvalScore:
    """
    One judge call, one dimension, one verdict.

    Args:
        model:     Any ChatModel — use a fast, cheap one; the judge follows
                   steps, it does not need to reason deeply.
        dimension: What is being measured ("correctness", "relevancy", ...).
        steps:     Explicit numbered grading steps — never a vague rubric.
        materials: The evidence, keyed by name ("question", "answer",
                   "reference", ...). Rendered as labeled sections.
    """
    body = "\n\n".join(f"## {name}\n{text}" for name, text in materials.items())
    messages = [
        Message(role="system", content=_JUDGE_SYSTEM),
        Message(role="user", content=f"# Grading steps\n{steps}\n\n{body}"),
    ]

    last_error: Exception | None = None
    for _ in range(2):  # one retry on a malformed verdict
        response = model.invoke(messages)
        if asyncio.iscoroutine(response):
            response = await response
        try:
            return _parse_verdict(dimension, response.content)
        except ValueError as error:
            last_error = error

    raise JudgeError(f"judge returned no usable verdict: {last_error}")


def _parse_verdict(dimension: str, content: str) -> EvalScore:
    """Parse the judge's JSON reply into an EvalScore. Raises ValueError."""
    text = content.strip()
    if text.startswith("```"):
        text = text.strip("`").removeprefix("json").strip()

    data = json.loads(text)  # ValueError on garbage
    if not isinstance(data, dict):
        raise ValueError(f"expected a JSON object, got {type(data).__name__}")
    if not isinstance(data.get("passed"), bool):
        raise ValueError("missing or non-boolean 'passed'")

    score = float(data.get("score", 0.0))
    return EvalScore(
        dimension=dimension,
        passed=data["passed"],
        score=min(1.0, max(0.0, score)),
        reason=str(data.get("reason", "")).strip(),
    )

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable, Any


@dataclass(frozen=True, slots=True)
class EvalScore:
    """One judgment on one dimension of one agent run."""

    dimension: str  # what was measured: "correctness", "faithfulness", ...
    passed: bool  # the binary verdict — gates and CI read this
    score: float  # 0.0 to 1.0 — dashboards and trends read this
    reason: str  # why — the first thing you read when it fails
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class EvalResult:
    """Every judgment made about a single agent run."""

    question: str
    answer: str
    scores: list[EvalScore] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """A run passes only if every dimension passes — no averaging."""
        return bool(self.scores) and all(s.passed for s in self.scores)


@runtime_checkable
class Evaluator(Protocol):
    """Eval contract: one agent run in, one judged result out."""

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
    ) -> EvalResult: ...

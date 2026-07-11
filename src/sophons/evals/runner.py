from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from sophons.evals.base import EvalScore, Evaluator
from sophons.evals.datasets import EvalCase, EvalDataset


@runtime_checkable
class RunnableAgent(Protocol):
    """What the runner needs from an agent: ask a question, get a result
    carrying the answer text and the tool calls that produced it."""

    async def run(self, input: str) -> Any: ...


@dataclass(frozen=True, slots=True)
class TrialResult:
    """One run of one case through the agent and every evaluator."""

    trial: int
    answer: str
    tool_calls: list[str]
    scores: list[EvalScore]
    skipped: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return bool(self.scores) and all(s.passed for s in self.scores)


@dataclass(frozen=True, slots=True)
class CaseResult:
    """Every trial of one case. Passes only if every trial passed —
    the pass^k rule: consistency, not luck."""

    case: EvalCase
    trials: list[TrialResult]

    @property
    def passed(self) -> bool:
        return bool(self.trials) and all(t.passed for t in self.trials)


@dataclass(frozen=True, slots=True)
class EvalRun:
    """The judged sweep of one agent over one dataset."""

    dataset_name: str
    dataset_version: str
    num_trials: int
    case_results: list[CaseResult]

    @property
    def pass_rate(self) -> float:
        """Fraction of individual trials that passed — the optimistic number."""
        trials = [t for c in self.case_results for t in c.trials]
        return sum(t.passed for t in trials) / len(trials) if trials else 0.0

    @property
    def pass_hat_k(self) -> float:
        """Fraction of cases that passed every trial (pass^k) — the honest
        number. Equal to pass_rate only when the agent is consistent."""
        cases = self.case_results
        return sum(c.passed for c in cases) / len(cases) if cases else 0.0

    def dimension_averages(self) -> dict[str, float]:
        """Mean score per dimension across all trials."""
        totals: dict[str, list[float]] = {}
        for case in self.case_results:
            for trial in case.trials:
                for score in trial.scores:
                    totals.setdefault(score.dimension, []).append(score.score)
        return {dim: sum(v) / len(v) for dim, v in sorted(totals.items())}

    def failures(self) -> list[tuple[EvalCase, TrialResult, EvalScore]]:
        """Every failing score, with the case and trial it came from."""
        found = []
        for case in self.case_results:
            for trial in case.trials:
                for score in trial.scores:
                    if not score.passed:
                        found.append((case.case, trial, score))
        return found


class EvalRunner:
    """
    Sweep an agent across a dataset, judging every run with every evaluator.

    Args:
        agent:      Anything with ``async run(input) -> result`` where the
                    result has ``.message`` and ``.tool_uses`` (sophons
                    ``Agent`` qualifies).
        evaluators: The judges to apply to every trial.
        num_trials: How many times to run each case. With ``num_trials > 1``
                    a case passes only if every trial passes — Sierra's
                    pass^k: agents are non-deterministic, and passing once
                    is weak evidence.

    An evaluator that raises ``ValueError`` for a case (missing context or
    expected_tools) is recorded as skipped for that trial, not failed —
    a case without ground truth is not a failing agent.
    """

    def __init__(
        self,
        agent: RunnableAgent,
        evaluators: list[Evaluator],
        *,
        num_trials: int = 1,
    ) -> None:
        self.agent = agent
        self.evaluators = evaluators
        self.num_trials = max(1, num_trials)

    async def run(self, dataset: EvalDataset) -> EvalRun:
        case_results = []
        for case in dataset.cases:
            trials = []
            for trial_index in range(self.num_trials):
                trials.append(await self._run_trial(case, trial_index))
            case_results.append(CaseResult(case=case, trials=trials))
        return EvalRun(
            dataset_name=dataset.name,
            dataset_version=dataset.version,
            num_trials=self.num_trials,
            case_results=case_results,
        )

    async def _run_trial(self, case: EvalCase, trial: int) -> TrialResult:
        result = await self.agent.run(case.question)
        answer: str = result.message
        tool_calls = [tool_use.name for tool_use in result.tool_uses]
        actual_tool_calls = [
            {"name": tool_use.name, "args": tool_use.input}
            for tool_use in result.tool_uses
        ]

        scores: list[EvalScore] = []
        skipped: list[str] = []
        for evaluator in self.evaluators:
            try:
                evaluated = await evaluator.evaluate(
                    case.question,
                    answer,
                    context=case.context,
                    reference=case.reference,
                    tool_calls=tool_calls,
                    expected_tools=case.expected_tools,
                    expected_tool_calls=case.expected_tool_calls,
                    actual_tool_calls=actual_tool_calls,
                )
            except ValueError:
                skipped.append(type(evaluator).__name__)
                continue
            scores.extend(evaluated.scores)

        return TrialResult(
            trial=trial,
            answer=answer,
            tool_calls=tool_calls,
            scores=scores,
            skipped=skipped,
        )

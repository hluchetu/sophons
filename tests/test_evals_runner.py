from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from sophons.evals import (
    EvalCase,
    EvalDataset,
    EvalRunner,
    TrajectoryEvaluator,
    render_report,
)


@dataclass
class FakeToolUse:
    name: str
    input: dict = field(default_factory=dict)


@dataclass
class FakeResult:
    message: str
    tool_uses: list[FakeToolUse]


class FlakyAgent:
    """Calls the expected tool only on even-numbered invocations."""

    def __init__(self) -> None:
        self.invocations = 0

    async def run(self, input: str) -> FakeResult:
        self.invocations += 1
        if self.invocations % 2 == 1:
            return FakeResult("answer", [FakeToolUse("search_docs")])
        return FakeResult("answer", [])  # forgot to search


class SteadyAgent:
    async def run(self, input: str) -> FakeResult:
        return FakeResult("answer", [FakeToolUse("search_docs")])


DATASET = EvalDataset(
    name="unit",
    version="v1",
    cases=[
        EvalCase(id="c1", question="q1", expected_tools=["search_docs"]),
        EvalCase(id="c2", question="q2", expected_tools=["search_docs"]),
    ],
)


@pytest.mark.asyncio
async def test_steady_agent_passes_all_trials() -> None:
    runner = EvalRunner(SteadyAgent(), [TrajectoryEvaluator()], num_trials=3)

    run = await runner.run(DATASET)

    assert run.pass_rate == 1.0
    assert run.pass_hat_k == 1.0
    assert run.dimension_averages() == {"trajectory": 1.0}
    assert run.failures() == []


@pytest.mark.asyncio
async def test_flaky_agent_exposed_by_pass_hat_k() -> None:
    runner = EvalRunner(FlakyAgent(), [TrajectoryEvaluator()], num_trials=2)

    run = await runner.run(DATASET)

    # Half the trials pass, but no case passes every trial.
    assert run.pass_rate == 0.5
    assert run.pass_hat_k == 0.0


@pytest.mark.asyncio
async def test_missing_ground_truth_is_skipped_not_failed() -> None:
    dataset = EvalDataset(
        name="unit",
        version="v1",
        cases=[EvalCase(id="c1", question="q1")],  # no expected_tools
    )
    runner = EvalRunner(SteadyAgent(), [TrajectoryEvaluator()])

    run = await runner.run(dataset)

    trial = run.case_results[0].trials[0]
    assert trial.scores == []
    assert trial.skipped == ["TrajectoryEvaluator"]
    assert trial.passed is False  # no scores means no pass — evals fail loud


@pytest.mark.asyncio
async def test_report_renders_rates_and_failures() -> None:
    runner = EvalRunner(FlakyAgent(), [TrajectoryEvaluator()], num_trials=2)

    report = render_report(await runner.run(DATASET))

    assert "pass rate (per trial):  50%" in report
    assert "pass^2 (per case):    0%" in report
    assert "trajectory" in report
    assert "[c1 trial" in report


def test_dataset_loads_from_json(tmp_path) -> None:
    path = tmp_path / "cases.json"
    path.write_text(
        '{"name": "d", "version": "v2", "cases": ['
        '{"id": "a", "question": "q", "expected_tools": ["t"],'
        ' "metadata": {"difficulty": "easy"}}]}'
    )

    dataset = EvalDataset.from_json(path)

    assert dataset.version == "v2"
    assert dataset.cases[0].expected_tools == ["t"]
    assert dataset.cases[0].metadata == {"difficulty": "easy"}

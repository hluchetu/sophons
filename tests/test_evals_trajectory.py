from __future__ import annotations

import pytest

from sophons.evals.trajectory import TrajectoryEvaluator


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mode", "expected", "actual", "want_passed", "want_score"),
    [
        # exact: same tools, same order, nothing extra
        ("exact", ["search"], ["search"], True, 1.0),
        ("exact", ["search"], ["search", "add"], False, 1.0),
        ("exact", ["add", "search"], ["search", "add"], False, 1.0),
        ("exact", ["search"], [], False, 0.0),
        # in-order: expected appears in order, gaps allowed
        ("in-order", ["search", "add"], ["search", "log", "add"], True, 1.0),
        ("in-order", ["add", "search"], ["search", "add"], False, 1.0),
        # any-order: multiset containment, counts respected
        ("any-order", ["add", "search"], ["search", "add"], True, 1.0),
        ("any-order", ["search", "search"], ["search"], False, 0.5),
        ("any-order", [], ["search"], True, 1.0),
    ],
)
async def test_trajectory_modes(
    mode: str,
    expected: list[str],
    actual: list[str],
    want_passed: bool,
    want_score: float,
) -> None:
    evaluator = TrajectoryEvaluator(mode=mode)  # type: ignore[arg-type]

    result = await evaluator.evaluate(
        "q", "a", tool_calls=actual, expected_tools=expected
    )

    assert len(result.scores) == 1
    score = result.scores[0]
    assert score.dimension == "trajectory"
    assert score.passed is want_passed
    assert score.score == pytest.approx(want_score)
    assert result.passed is want_passed


@pytest.mark.asyncio
async def test_missing_expected_tools_raises() -> None:
    evaluator = TrajectoryEvaluator()

    with pytest.raises(ValueError, match="expected_tools"):
        await evaluator.evaluate("q", "a", tool_calls=["search"])


@pytest.mark.asyncio
async def test_no_tool_calls_defaults_to_empty() -> None:
    evaluator = TrajectoryEvaluator(mode="any-order")

    result = await evaluator.evaluate("q", "a", expected_tools=[])

    assert result.passed is True

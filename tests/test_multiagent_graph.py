from __future__ import annotations

import asyncio
from typing import Any

import pytest

from sophons.agents import AgentMetrics, AgentResult, StopReason
from sophons.multiagent import GraphBuilder, Status


class FakeAgent:
    def __init__(
        self,
        name: str,
        *,
        fail: bool = False,
        calls: list[tuple[str, str]] | None = None,
    ) -> None:
        self.name = name
        self.fail = fail
        self.calls = calls if calls is not None else []

    async def run(self, input: str) -> AgentResult:
        self.calls.append((self.name, input))
        if self.fail:
            raise RuntimeError(f"{self.name} failed")

        return AgentResult(
            stop_reason=StopReason.END_TURN,
            message=f"{self.name} output",
            metrics=AgentMetrics(model_calls=1, input_tokens=2, output_tokens=3),
            tool_uses=[],
            tool_results=[],
            success=True,
        )


def add_fake_node(
    builder: GraphBuilder,
    agent: FakeAgent,
    node_id: str | None = None,
    **kwargs: Any,
) -> Any:
    """Keep test doubles local while the public API stays Agent-typed."""

    return builder.add_node(agent, node_id, **kwargs)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_graph_runs_nodes_in_dependency_order() -> None:
    calls: list[tuple[str, str]] = []
    builder = GraphBuilder()
    researcher = add_fake_node(builder, FakeAgent("researcher", calls=calls))
    writer = add_fake_node(builder, FakeAgent("writer", calls=calls))
    builder.add_edge(researcher, writer)

    result = await builder.build().run("Explain graph orchestration")

    assert result.status is Status.COMPLETED
    assert result.execution_order == ["researcher", "writer"]
    assert result.completed_node_ids == ["researcher", "writer"]
    assert result.accumulated_metrics.model_calls == 2
    assert calls[0] == ("researcher", "Explain graph orchestration")
    assert "researcher output" in calls[1][1]


@pytest.mark.asyncio
async def test_graph_executes_ready_nodes_in_parallel() -> None:
    started: list[str] = []
    both_started = asyncio.Event()

    class ParallelAgent(FakeAgent):
        async def run(self, input: str) -> AgentResult:
            started.append(self.name)
            if len(started) == 2:
                both_started.set()
            await asyncio.wait_for(both_started.wait(), timeout=0.2)
            return await super().run(input)

    builder = GraphBuilder()
    add_fake_node(builder, ParallelAgent("left"))
    add_fake_node(builder, ParallelAgent("right"))

    result = await builder.build().run("parallel task")

    assert result.success
    assert set(started) == {"left", "right"}


@pytest.mark.asyncio
async def test_graph_uses_conditional_edges() -> None:
    builder = GraphBuilder()
    router = add_fake_node(builder, FakeAgent("router"))
    approved = add_fake_node(builder, FakeAgent("approved"))
    rejected = add_fake_node(builder, FakeAgent("rejected"))
    builder.add_edge(
        router,
        approved,
        lambda state, *, invocation_state: invocation_state["approved"],
    )
    builder.add_edge(
        router,
        rejected,
        lambda state, *, invocation_state: not invocation_state["approved"],
    )

    graph = builder.build()
    result = await graph.run(
        "route this",
        invocation_state={"approved": True},
    )

    assert result.success
    assert result.execution_order == ["router", "approved"]
    assert "rejected" not in result.results


@pytest.mark.asyncio
async def test_cyclic_graph_stops_at_execution_limit() -> None:
    builder = GraphBuilder()
    first = add_fake_node(builder, FakeAgent("first"))
    second = add_fake_node(builder, FakeAgent("second"))
    builder.add_edge(first, second)
    builder.add_edge(second, first)
    builder.set_entry_point(first)
    builder.set_max_node_executions(3)

    result = await builder.build().run("iterate")

    assert result.status is Status.FAILED
    assert result.execution_order == ["first", "second", "first"]
    assert result.execution_count == 3


@pytest.mark.asyncio
async def test_graph_can_use_another_graph_as_a_node() -> None:
    inner_builder = GraphBuilder().set_graph_id("inner")
    add_fake_node(inner_builder, FakeAgent("researcher"))
    inner_graph = inner_builder.build()

    outer_builder = GraphBuilder().set_graph_id("outer")
    inner_node = outer_builder.add_node(inner_graph)
    writer = add_fake_node(outer_builder, FakeAgent("writer"))
    outer_builder.add_edge(inner_node, writer)

    result = await outer_builder.build().run("nested task")

    assert result.success
    assert result.execution_order == ["inner", "writer"]
    assert result.execution_count == 2


@pytest.mark.asyncio
async def test_graph_records_node_failure() -> None:
    builder = GraphBuilder()
    add_fake_node(builder, FakeAgent("broken", fail=True))

    result = await builder.build().run("fail")

    assert result.status is Status.FAILED
    assert result.failed_node_ids == ["broken"]
    assert isinstance(result.results["broken"].result, RuntimeError)


def test_builder_rejects_duplicate_executor_instances() -> None:
    builder = GraphBuilder()
    agent = FakeAgent("shared")
    add_fake_node(builder, agent, "first")

    with pytest.raises(ValueError, match="unique executor"):
        add_fake_node(builder, agent, "second")


def test_builder_requires_entry_point_for_a_cycle() -> None:
    builder = GraphBuilder()
    first = add_fake_node(builder, FakeAgent("first"))
    second = add_fake_node(builder, FakeAgent("second"))
    builder.add_edge(first, second)
    builder.add_edge(second, first)

    with pytest.raises(ValueError, match="entry point"):
        builder.build()


def test_builder_rejects_an_unbounded_cycle() -> None:
    builder = GraphBuilder()
    first = add_fake_node(builder, FakeAgent("first"))
    second = add_fake_node(builder, FakeAgent("second"))
    builder.add_edge(first, second)
    builder.add_edge(second, first)
    builder.set_entry_point(first)

    with pytest.raises(ValueError, match="require max node executions"):
        builder.build()


@pytest.mark.asyncio
async def test_all_join_waits_for_every_active_predecessor() -> None:
    calls: list[tuple[str, str]] = []
    builder = GraphBuilder()
    start = add_fake_node(builder, FakeAgent("start", calls=calls))
    left = add_fake_node(builder, FakeAgent("left", calls=calls))
    right = add_fake_node(builder, FakeAgent("right", calls=calls))
    report = add_fake_node(builder, FakeAgent("report", calls=calls))
    builder.add_edge(start, left)
    builder.add_edge(start, right)
    builder.add_edge(left, report)
    builder.add_edge(right, report)

    result = await builder.build().run("fan in")

    assert result.success
    assert result.execution_order == ["start", "left", "right", "report"]
    report_input = next(value for name, value in calls if name == "report")
    assert "left output" in report_input
    assert "right output" in report_input


@pytest.mark.asyncio
async def test_inactive_conditional_edge_does_not_block_all_join() -> None:
    builder = GraphBuilder()
    start = add_fake_node(builder, FakeAgent("start"))
    active = add_fake_node(builder, FakeAgent("active"))
    inactive = add_fake_node(builder, FakeAgent("inactive"))
    report = add_fake_node(builder, FakeAgent("report"))
    builder.add_edge(start, active)
    builder.add_edge(start, inactive, lambda state: False)
    builder.add_edge(active, report)
    builder.add_edge(inactive, report)

    result = await builder.build().run("conditional fan in")

    assert result.success
    assert result.execution_order == ["start", "active", "report"]
    assert result.skipped_nodes == ["inactive"]


@pytest.mark.asyncio
async def test_any_join_runs_once_for_a_parallel_batch() -> None:
    builder = GraphBuilder()
    start = add_fake_node(builder, FakeAgent("start"))
    left = add_fake_node(builder, FakeAgent("left"))
    right = add_fake_node(builder, FakeAgent("right"))
    first_answer = add_fake_node(
        builder,
        FakeAgent("first_answer"),
        join="any",
    )
    builder.add_edge(start, left)
    builder.add_edge(start, right)
    builder.add_edge(left, first_answer)
    builder.add_edge(right, first_answer)

    result = await builder.build().run("first result")

    assert result.success
    assert result.execution_order.count("first_answer") == 1


@pytest.mark.asyncio
async def test_graph_preserves_revisit_history() -> None:
    builder = GraphBuilder()
    first = add_fake_node(builder, FakeAgent("first"))
    second = add_fake_node(builder, FakeAgent("second"))
    builder.add_edge(first, second)
    builder.add_edge(second, first)
    builder.set_entry_point(first)
    builder.set_max_node_executions(3)

    result = await builder.build().run("review")

    assert [item.execution_index for item in result.node_history["first"]] == [1, 3]
    assert len(result.node_history["second"]) == 1


@pytest.mark.asyncio
async def test_terminal_output_is_deterministic() -> None:
    builder = GraphBuilder()
    add_fake_node(builder, FakeAgent("alpha"))
    add_fake_node(builder, FakeAgent("beta"))

    result = await builder.build().run("two terminals")

    assert result.output == {
        "alpha": "alpha output",
        "beta": "beta output",
    }


@pytest.mark.asyncio
async def test_concurrent_runs_keep_invocation_state_isolated() -> None:
    builder = GraphBuilder()
    router = add_fake_node(builder, FakeAgent("router"))
    left = add_fake_node(builder, FakeAgent("left"))
    right = add_fake_node(builder, FakeAgent("right"))
    builder.add_edge(
        router,
        left,
        lambda state, *, invocation_state: invocation_state["route"] == "left",
    )
    builder.add_edge(
        router,
        right,
        lambda state, *, invocation_state: invocation_state["route"] == "right",
    )
    graph = builder.build()

    left_result, right_result = await asyncio.gather(
        graph.run("one", invocation_state={"route": "left"}),
        graph.run("two", invocation_state={"route": "right"}),
    )

    assert left_result.execution_order == ["router", "left"]
    assert right_result.execution_order == ["router", "right"]

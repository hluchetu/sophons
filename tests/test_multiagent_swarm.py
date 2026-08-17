from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from sophons.agents import AgentMetrics, AgentResult, StopReason
from sophons.multiagent import GraphBuilder, Status, Swarm


class ScriptedAgent:
    def __init__(
        self,
        name: str,
        handoffs: list[tuple[str, str, dict[str, Any]]] | None = None,
        *,
        fail: bool = False,
    ) -> None:
        self.name = name
        self.handoffs = list(handoffs or [])
        self.fail = fail
        self.prompts: list[str] = []
        self.tool_responses: list[dict[str, Any]] = []
        self._loop = SimpleNamespace(_tools={})

    async def run(self, input: str) -> AgentResult:
        self.prompts.append(input)
        if self.fail:
            raise RuntimeError(f"{self.name} failed")

        if self.handoffs:
            target, message, context = self.handoffs.pop(0)
            response = self._loop._tools["handoff_to_agent"].call(
                {
                    "agent_name": target,
                    "message": message,
                    "context": context,
                }
            )
            self.tool_responses.append(response)

        return AgentResult(
            stop_reason=StopReason.END_TURN,
            message=f"{self.name} completed",
            metrics=AgentMetrics(model_calls=1, input_tokens=2, output_tokens=3),
            tool_uses=[],
            tool_results=[],
            success=True,
        )


def make_swarm(agents: list[ScriptedAgent], **kwargs: Any) -> Swarm:
    return Swarm(agents, **kwargs)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_entry_agent_can_complete_without_handoff() -> None:
    triage = ScriptedAgent("triage")

    result = await make_swarm([triage]).run("Help me")

    assert result.status is Status.COMPLETED
    assert result.node_history == ["triage"]
    assert result.handoff_count == 0
    assert result.output == "triage completed"


@pytest.mark.asyncio
async def test_swarm_commits_multiple_structured_handoffs() -> None:
    triage = ScriptedAgent(
        "triage",
        [("billing", "Check the invoice", {"invoice": "INV-7"})],
    )
    billing = ScriptedAgent(
        "billing",
        [("fraud", "Review the charge", {"risk": "high"})],
    )
    fraud = ScriptedAgent("fraud")
    swarm = make_swarm(
        [triage, billing, fraud],
        allowed_handoffs={
            "triage": {"billing"},
            "billing": {"fraud"},
            "fraud": set(),
        },
    )

    result = await swarm.run("Unknown card charge")

    assert result.success
    assert result.node_history == ["triage", "billing", "fraud"]
    assert [(item.source, item.target) for item in result.handoff_history] == [
        ("triage", "billing"),
        ("billing", "fraud"),
    ]
    assert "INV-7" in billing.prompts[0]
    assert "high" in fraud.prompts[0]
    assert result.output == "fraud completed"


@pytest.mark.asyncio
async def test_swarm_rejects_disallowed_handoff() -> None:
    triage = ScriptedAgent("triage", [("fraud", "Skip billing", {})])
    billing = ScriptedAgent("billing")
    fraud = ScriptedAgent("fraud")
    swarm = make_swarm(
        [triage, billing, fraud],
        allowed_handoffs={
            "triage": {"billing"},
            "billing": set(),
            "fraud": set(),
        },
    )

    result = await swarm.run("charge")

    assert result.status is Status.FAILED
    assert "cannot hand off" in (result.stop_reason or "")
    assert triage.tool_responses[0]["status"] == "error"
    assert result.handoff_count == 0


@pytest.mark.asyncio
async def test_swarm_rejects_unknown_handoff_target() -> None:
    triage = ScriptedAgent("triage", [("ghost", "Find them", {})])

    result = await make_swarm([triage]).run("task")

    assert result.status is Status.FAILED
    assert "ghost" in (result.stop_reason or "")


@pytest.mark.asyncio
async def test_swarm_enforces_maximum_handoffs() -> None:
    first = ScriptedAgent("first", [("second", "go", {})])
    second = ScriptedAgent("second", [("first", "back", {})])
    swarm = make_swarm(
        [first, second],
        max_handoffs=1,
        repetitive_handoff_window=0,
    )

    result = await swarm.run("loop")

    assert result.status is Status.FAILED
    assert result.handoff_count == 1
    assert result.stop_reason == "Maximum handoffs reached: 1"


@pytest.mark.asyncio
async def test_swarm_enforces_maximum_iterations() -> None:
    first = ScriptedAgent("first", [("second", "go", {}), ("second", "again", {})])
    second = ScriptedAgent("second", [("first", "back", {})])
    swarm = make_swarm(
        [first, second],
        max_iterations=2,
        repetitive_handoff_window=0,
    )

    result = await swarm.run("loop")

    assert result.status is Status.FAILED
    assert result.iteration_count == 2
    assert result.stop_reason == "Maximum iterations reached: 2"


@pytest.mark.asyncio
async def test_swarm_detects_repetitive_handoffs() -> None:
    first = ScriptedAgent(
        "first",
        [("second", "go", {}) for _ in range(4)],
    )
    second = ScriptedAgent(
        "second",
        [("first", "back", {}) for _ in range(4)],
    )
    swarm = make_swarm(
        [first, second],
        max_iterations=20,
        repetitive_handoff_window=4,
        repetitive_handoff_min_unique_agents=3,
    )

    result = await swarm.run("loop")

    assert result.status is Status.FAILED
    assert result.node_history == ["first", "second", "first", "second"]
    assert "Repetitive handoff pattern" in (result.stop_reason or "")


@pytest.mark.asyncio
async def test_swarm_records_active_agent_failure() -> None:
    broken = ScriptedAgent("broken", fail=True)

    result = await make_swarm([broken]).run("fail")

    assert result.status is Status.FAILED
    assert isinstance(result.results["broken"].result, RuntimeError)
    assert result.stop_reason == "broken failed"


@pytest.mark.asyncio
async def test_shared_context_discards_oldest_entries() -> None:
    first = ScriptedAgent(
        "first",
        [("second", "one", {"step": 1}), ("second", "three", {"step": 3})],
    )
    second = ScriptedAgent(
        "second",
        [("first", "two", {"step": 2})],
    )
    swarm = make_swarm(
        [first, second],
        max_context_entries=2,
        repetitive_handoff_window=0,
    )

    result = await swarm.run("context")

    assert result.success
    assert [entry["values"]["step"] for entry in result.shared_context.entries] == [2, 3]


@pytest.mark.asyncio
async def test_swarm_can_run_as_a_graph_node() -> None:
    triage = ScriptedAgent("triage", [("specialist", "continue", {})])
    specialist = ScriptedAgent("specialist")
    swarm = make_swarm([triage, specialist])
    writer = ScriptedAgent("writer")
    builder = GraphBuilder()
    swarm_node = builder.add_node(swarm, "support_team")
    writer_node = builder.add_node(writer, "writer")  # type: ignore[arg-type]
    builder.add_edge(swarm_node, writer_node)

    result = await builder.build().run("resolve ticket")

    assert result.success
    assert result.execution_order == ["support_team", "writer"]
    assert "specialist completed" in writer.prompts[0]

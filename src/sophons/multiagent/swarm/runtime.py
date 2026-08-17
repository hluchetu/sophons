"""Runtime for bounded, tool-directed Sophons swarms."""

from __future__ import annotations

import asyncio
import builtins
import json
import time
from contextvars import ContextVar
from typing import Any

from sophons.agents import Agent, AgentMetrics
from sophons.multiagent.base import MultiAgentBase, NodeResult, Status
from sophons.multiagent.swarm.handoff import HandoffTool
from sophons.multiagent.swarm.types import (
    Handoff,
    HandoffRecord,
    SharedContext,
    SwarmNode,
    SwarmResult,
    SwarmState,
)


class Swarm(MultiAgentBase):
    """A bounded team whose active agent chooses the next specialist."""

    def __init__(
        self,
        agents: list[Agent],
        *,
        entry_point: Agent | str | None = None,
        descriptions: dict[str, str] | None = None,
        allowed_handoffs: dict[str, set[str]] | None = None,
        max_handoffs: int = 20,
        max_iterations: int = 20,
        execution_timeout: float = 900.0,
        node_timeout: float = 300.0,
        repetitive_handoff_window: int = 6,
        repetitive_handoff_min_unique_agents: int = 3,
        max_context_entries: int = 20,
        id: str = "default_swarm",
    ) -> None:
        super().__init__(id=id)
        if not agents:
            raise ValueError("Swarm must contain at least one agent.")
        if max_handoffs < 0 or max_iterations < 1:
            raise ValueError("Swarm limits must be positive.")
        if execution_timeout <= 0 or node_timeout <= 0:
            raise ValueError("Swarm timeouts must be greater than zero.")

        names = [self._agent_name(agent, index) for index, agent in enumerate(agents)]
        if len(set(names)) != len(names):
            raise ValueError("Every swarm agent must have a unique name.")
        if len({builtins.id(agent) for agent in agents}) != len(agents):
            raise ValueError("Every swarm member must use a unique agent instance.")

        all_names = set(names)
        configured = allowed_handoffs or {}
        unknown_sources = set(configured) - all_names
        unknown_targets = set().union(*configured.values()) - all_names if configured else set()
        if unknown_sources or unknown_targets:
            raise ValueError("Allowed handoffs contain unknown agent names.")

        descriptions = descriptions or {}
        self.nodes = {
            name: SwarmNode(
                node_id=name,
                agent=agent,
                description=descriptions.get(name, ""),
                allowed_handoffs=frozenset(
                    configured.get(name, all_names - {name})
                ),
            )
            for name, agent in zip(names, agents, strict=True)
        }
        self.entry_agent = self._resolve_entry_point(entry_point, agents, names)
        self.max_handoffs = max_handoffs
        self.max_iterations = max_iterations
        self.execution_timeout = execution_timeout
        self.node_timeout = node_timeout
        self.repetitive_handoff_window = repetitive_handoff_window
        self.repetitive_handoff_min_unique_agents = repetitive_handoff_min_unique_agents
        self.max_context_entries = max_context_entries
        self._active_node: ContextVar[str | None] = ContextVar(
            f"{id}:active_node", default=None
        )
        self._pending_handoff: ContextVar[Handoff | None] = ContextVar(
            f"{id}:pending_handoff", default=None
        )
        self._handoff_error: ContextVar[str | None] = ContextVar(
            f"{id}:handoff_error", default=None
        )
        self._inject_handoff_tools()

    async def run(
        self,
        input: str,
        *,
        invocation_state: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> SwarmResult:
        del invocation_state, kwargs
        state = SwarmState(
            task=input,
            current_node_id=self.entry_agent,
            status=Status.EXECUTING,
            shared_context=SharedContext(max_entries=self.max_context_entries),
        )
        started = time.perf_counter()

        while state.status is Status.EXECUTING:
            reason = self._limit_reason(state, started)
            if reason is not None:
                state.status = Status.FAILED
                state.stop_reason = reason
                break

            node = self.nodes[state.current_node_id]
            self._active_node.set(node.node_id)
            self._pending_handoff.set(None)
            self._handoff_error.set(None)
            prompt = self._build_node_input(state, node)
            turn_started = time.perf_counter()

            try:
                remaining = self.execution_timeout - (time.perf_counter() - started)
                agent_result = await asyncio.wait_for(
                    node.agent.run(prompt),
                    timeout=min(self.node_timeout, max(remaining, 0.001)),
                )
                status = Status.COMPLETED if agent_result.success else Status.FAILED
                node_result = NodeResult(
                    result=agent_result,
                    execution_time=round((time.perf_counter() - turn_started) * 1000),
                    status=status,
                    accumulated_metrics=agent_result.metrics,
                    execution_count=1,
                )
            except Exception as error:
                node_result = NodeResult(
                    result=error,
                    execution_time=round((time.perf_counter() - turn_started) * 1000),
                    status=Status.FAILED,
                    execution_count=1,
                )

            state.results[node.node_id] = node_result
            state.node_history.append(node.node_id)
            state.execution_count += 1
            self._accumulate_metrics(state.accumulated_metrics, node_result.accumulated_metrics)

            if node_result.status is Status.FAILED:
                state.status = Status.FAILED
                state.stop_reason = str(node_result.result)
                break

            if error := self._handoff_error.get():
                state.status = Status.FAILED
                state.stop_reason = error
                break

            handoff = self._pending_handoff.get()
            if handoff is None:
                state.status = Status.COMPLETED
                state.stop_reason = "completed"
                break

            if len(state.handoff_history) >= self.max_handoffs:
                state.status = Status.FAILED
                state.stop_reason = f"Maximum handoffs reached: {self.max_handoffs}"
                break

            if handoff.context:
                state.shared_context.add(node.node_id, handoff.context)
            state.handoff_history.append(
                HandoffRecord(
                    source=node.node_id,
                    target=handoff.target,
                    message=handoff.message,
                    reason=handoff.reason,
                )
            )
            state.current_node_id = handoff.target

        state.execution_time = round((time.perf_counter() - started) * 1000)
        return self._build_result(state)

    def _request_handoff(self, handoff: Handoff) -> None:
        source = self._active_node.get()
        if source is None:
            raise ValueError("Handoff requested outside an active swarm turn.")
        if handoff.target not in self.nodes[source].allowed_handoffs:
            message = f"Agent {source!r} cannot hand off to {handoff.target!r}."
            self._handoff_error.set(message)
            raise ValueError(message)
        if self._pending_handoff.get() is not None:
            message = "An agent can request only one handoff per turn."
            self._handoff_error.set(message)
            raise ValueError(message)
        self._pending_handoff.set(handoff)

    def _inject_handoff_tools(self) -> None:
        tool = HandoffTool(self._request_handoff)
        for node in self.nodes.values():
            tools = node.agent._loop._tools
            if tool.name in tools:
                raise ValueError(
                    f"Agent {node.node_id!r} already has a {tool.name!r} tool."
                )
            tools[tool.name] = tool

    def _build_node_input(self, state: SwarmState, node: SwarmNode) -> str:
        sections = [f"User request:\n{state.task}"]
        if state.handoff_history:
            latest = state.handoff_history[-1]
            sections.append(f"Handoff from {latest.source}:\n{latest.message}")
        if state.shared_context.entries:
            sections.append(
                "Shared knowledge:\n"
                + json.dumps(state.shared_context.entries, ensure_ascii=False)
            )

        peers = [
            f"- {peer.node_id}: {peer.description or 'No description provided.'}"
            for peer in self.nodes.values()
            if peer.node_id in node.allowed_handoffs
        ]
        if peers:
            sections.append(
                "Available handoff targets:\n"
                + "\n".join(peers)
                + "\nUse handoff_to_agent only if another specialist should continue."
            )
        return "\n\n".join(sections)

    def _limit_reason(self, state: SwarmState, started: float) -> str | None:
        if state.execution_count >= self.max_iterations:
            return f"Maximum iterations reached: {self.max_iterations}"
        if time.perf_counter() - started >= self.execution_timeout:
            return f"Execution timed out after {self.execution_timeout}s"

        window = self.repetitive_handoff_window
        if window > 0 and len(state.node_history) >= window:
            recent = state.node_history[-window:]
            if len(set(recent)) < self.repetitive_handoff_min_unique_agents:
                return f"Repetitive handoff pattern detected: {' -> '.join(recent)}"
        return None

    @staticmethod
    def _accumulate_metrics(total: AgentMetrics, metrics: AgentMetrics) -> None:
        for name in (
            "steps", "model_calls", "tool_calls", "input_tokens", "output_tokens",
            "cache_read_tokens", "cache_write_tokens", "duration_ms",
        ):
            setattr(total, name, getattr(total, name) + getattr(metrics, name))
        for name, stats in metrics.per_tool.items():
            current = total.per_tool.setdefault(name, type(stats)())
            current.calls += stats.calls
            current.errors += stats.errors
            current.total_ms += stats.total_ms

    def _build_result(self, state: SwarmState) -> SwarmResult:
        last = state.node_history[-1] if state.node_history else self.entry_agent
        return SwarmResult(
            status=state.status,
            results=dict(state.results),
            accumulated_metrics=state.accumulated_metrics,
            execution_count=state.execution_count,
            execution_time=state.execution_time,
            entry_agent=self.entry_agent,
            last_active_agent=last,
            node_history=list(state.node_history),
            handoff_history=list(state.handoff_history),
            shared_context=state.shared_context,
            iteration_count=state.execution_count,
            handoff_count=len(state.handoff_history),
            stop_reason=state.stop_reason,
        )

    def _resolve_entry_point(
        self,
        entry_point: Agent | str | None,
        agents: list[Agent],
        names: list[str],
    ) -> str:
        if entry_point is None:
            return names[0]
        if isinstance(entry_point, str):
            if entry_point not in self.nodes:
                raise ValueError(f"Entry agent {entry_point!r} is not in the swarm.")
            return entry_point
        for name, agent in zip(names, agents, strict=True):
            if agent is entry_point:
                return name
        raise ValueError("Entry agent is not a member of the swarm.")

    @staticmethod
    def _agent_name(agent: Agent, index: int) -> str:
        name = getattr(agent, "name", None)
        return str(name) if name else f"node_{index}"

    def serialize_state(self) -> dict[str, Any]:
        raise NotImplementedError(
            "Swarm resume requires AgentResult serialization, planned for phase 5."
        )

    def deserialize_state(self, payload: dict[str, Any]) -> None:
        del payload
        raise NotImplementedError(
            "Swarm resume requires AgentResult serialization, planned for phase 5."
        )

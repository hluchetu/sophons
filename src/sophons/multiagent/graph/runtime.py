"""Invocation-local scheduler for Sophons directed graphs."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Iterable
from typing import Any

from sophons.agents import Agent, AgentMetrics, AgentResult
from sophons.multiagent.base import MultiAgentBase, MultiAgentResult, NodeResult, Status
from sophons.multiagent.graph.types import (
    GraphEdge,
    GraphNode,
    GraphResult,
    GraphState,
    NodeExecution,
)


class Graph(MultiAgentBase):
    """Execute a validated graph using deterministic batch scheduling."""

    def __init__(
        self,
        *,
        nodes: dict[str, GraphNode],
        edges: frozenset[GraphEdge],
        entry_points: frozenset[str],
        max_node_executions: int | None = None,
        execution_timeout: float | None = None,
        node_timeout: float | None = None,
        max_concurrency: int | None = None,
        id: str = "default_graph",
    ) -> None:
        super().__init__(id=id)
        self.nodes = nodes
        self.edges = edges
        self.entry_points = entry_points
        self.max_node_executions = max_node_executions
        self.execution_timeout = execution_timeout
        self.node_timeout = node_timeout
        self.max_concurrency = max_concurrency
        self._incoming = self._group_edges("target")
        self._outgoing = self._group_edges("source")

    async def run(
        self,
        input: str,
        *,
        invocation_state: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> GraphResult:
        del kwargs
        state = GraphState(
            task=input,
            invocation_state=dict(invocation_state or {}),
            status=Status.EXECUTING,
        )
        started = time.perf_counter()
        ready = sorted(self.entry_points)

        while ready and state.status is Status.EXECUTING:
            limit_error = self._limit_error(state, started)
            if limit_error:
                state.status = Status.FAILED
                state.error = limit_error
                break

            if self.max_node_executions is not None:
                remaining = self.max_node_executions - state.execution_count
                ready = ready[:remaining]

            self._consume_signals(state, ready)
            results = await self._execute_batch(state, ready, started)

            for node_id, result in zip(ready, results, strict=True):
                self._record_result(state, node_id, result)

            if state.failed_nodes:
                state.status = Status.FAILED
                failed = sorted(state.failed_nodes)
                state.error = f"Graph node failure: {', '.join(failed)}"
                break
            if state.interrupted_nodes:
                state.status = Status.INTERRUPTED
                break

            self._evaluate_edges(state, ready)
            self._resolve_skipped_nodes(state)
            ready = self._find_ready_nodes(state)

        if state.status is Status.EXECUTING:
            state.status = Status.COMPLETED

        state.execution_time = round((time.perf_counter() - started) * 1000)
        return self._build_result(state)

    async def _execute_batch(
        self,
        state: GraphState,
        node_ids: list[str],
        graph_started: float,
    ) -> list[NodeResult]:
        semaphore = asyncio.Semaphore(self.max_concurrency or len(node_ids))

        async def execute(node_id: str) -> NodeResult:
            async with semaphore:
                return await self._execute_node(state, self.nodes[node_id], graph_started)

        return list(await asyncio.gather(*(execute(node_id) for node_id in node_ids)))

    async def _execute_node(
        self,
        state: GraphState,
        node: GraphNode,
        graph_started: float,
    ) -> NodeResult:
        started = time.perf_counter()
        try:
            node_input = self._build_node_input(state, node)
            timeout = node.timeout or self.node_timeout
            if self.execution_timeout is not None:
                remaining = self.execution_timeout - (time.perf_counter() - graph_started)
                timeout = min(timeout, remaining) if timeout is not None else remaining

            invocation = self._invoke_executor(node, node_input, state.invocation_state)
            result = await asyncio.wait_for(invocation, timeout=max(timeout, 0.001)) if timeout else await invocation

            if isinstance(result, MultiAgentResult):
                status = result.status
                metrics = result.accumulated_metrics
                interrupts = result.interrupts
                nested_count = result.execution_count
            else:
                status = Status.COMPLETED if result.success else Status.FAILED
                metrics = result.metrics
                interrupts = []
                nested_count = 1

            return NodeResult(
                result=result,
                execution_time=round((time.perf_counter() - started) * 1000),
                status=status,
                accumulated_metrics=metrics,
                execution_count=nested_count,
                interrupts=interrupts,
            )
        except Exception as error:
            return NodeResult(
                result=error,
                execution_time=round((time.perf_counter() - started) * 1000),
                status=Status.FAILED,
                execution_count=1,
            )

    async def _invoke_executor(
        self,
        node: GraphNode,
        node_input: str,
        invocation_state: dict[str, Any],
    ) -> AgentResult | MultiAgentResult:
        if isinstance(node.executor, MultiAgentBase):
            return await node.executor.run(
                node_input,
                invocation_state=invocation_state,
            )
        return await node.executor.run(node_input)

    def _record_result(
        self,
        state: GraphState,
        node_id: str,
        result: NodeResult,
    ) -> None:
        index = state.execution_count + 1
        execution = NodeExecution(node_id, index, result)
        state.latest_results[node_id] = result
        state.node_history.setdefault(node_id, []).append(execution)
        state.execution_order.append(node_id)
        state.execution_count = index
        self._accumulate_metrics(state.accumulated_metrics, result.accumulated_metrics)

        if result.status is Status.FAILED:
            state.failed_nodes.add(node_id)
        elif result.status is Status.INTERRUPTED:
            state.interrupted_nodes.add(node_id)

    def _evaluate_edges(self, state: GraphState, source_ids: Iterable[str]) -> None:
        for source_id in source_ids:
            for edge in self._outgoing.get(source_id, ()):
                state.edge_values[(edge.source, edge.target)] = edge.should_traverse(
                    state,
                    invocation_state=state.invocation_state,
                )

    def _find_ready_nodes(self, state: GraphState) -> list[str]:
        ready: list[str] = []
        for node_id, node in self.nodes.items():
            incoming = self._incoming.get(node_id, ())
            if not incoming or node_id in state.failed_nodes:
                continue

            active: list[GraphEdge] = []
            unresolved = False
            for edge in incoming:
                if edge.source in state.skipped_nodes:
                    continue
                if edge.source not in state.node_history:
                    unresolved = True
                    continue
                if state.edge_values.get((edge.source, edge.target), False):
                    active.append(edge)

            if node.join == "all" and unresolved:
                continue
            if not active:
                continue

            consumed = state.consumed_signals.get(node_id, {})
            fresh = [
                edge
                for edge in active
                if state.execution_count_for(edge.source)
                > consumed.get(edge.source, 0)
            ]
            if node.join == "any" and fresh:
                ready.append(node_id)
            elif node.join == "all" and len(fresh) == len(active):
                ready.append(node_id)
        return sorted(ready)

    def _consume_signals(self, state: GraphState, node_ids: Iterable[str]) -> None:
        for node_id in node_ids:
            consumed = state.consumed_signals.setdefault(node_id, {})
            for edge in self._incoming.get(node_id, ()):
                if state.edge_values.get((edge.source, edge.target), False):
                    consumed[edge.source] = state.execution_count_for(edge.source)

    def _resolve_skipped_nodes(self, state: GraphState) -> None:
        changed = True
        while changed:
            changed = False
            for node_id in self.nodes:
                if node_id in state.node_history or node_id in state.skipped_nodes:
                    continue
                incoming = self._incoming.get(node_id, ())
                if not incoming or node_id in self.entry_points:
                    continue
                sources_resolved = all(
                    edge.source in state.node_history
                    or edge.source in state.skipped_nodes
                    for edge in incoming
                )
                has_active_edge = any(
                    state.edge_values.get((edge.source, edge.target), False)
                    for edge in incoming
                )
                if sources_resolved and not has_active_edge:
                    state.skipped_nodes.add(node_id)
                    changed = True

    def _build_node_input(self, state: GraphState, node: GraphNode) -> str:
        if node.input_mapper is not None:
            return self._stringify_input(node.input_mapper(state))

        incoming = [
            edge
            for edge in self._incoming.get(node.node_id, ())
            if state.edge_values.get((edge.source, edge.target), False)
            and edge.source in state.latest_results
        ]
        if not incoming:
            return state.task

        sections = [f"Original task:\n{state.task}", "Inputs from previous nodes:"]
        for edge in sorted(incoming, key=lambda item: item.source):
            sections.append(f"From {edge.source}:\n{state.latest_results[edge.source]}")
        return "\n\n".join(sections)

    @staticmethod
    def _stringify_input(value: Any) -> str:
        if isinstance(value, str):
            return value
        model_dump_json = getattr(value, "model_dump_json", None)
        if callable(model_dump_json):
            return str(model_dump_json())
        try:
            return json.dumps(value)
        except TypeError:
            return str(value)

    def _limit_error(self, state: GraphState, started: float) -> str | None:
        if (
            self.max_node_executions is not None
            and state.execution_count >= self.max_node_executions
        ):
            return f"Maximum node executions reached: {self.max_node_executions}"
        if (
            self.execution_timeout is not None
            and time.perf_counter() - started >= self.execution_timeout
        ):
            return f"Graph execution timed out after {self.execution_timeout}s"
        return None

    def _build_result(self, state: GraphState) -> GraphResult:
        terminals = sorted(set(self.nodes) - set(self._outgoing))
        interrupts = [
            interrupt
            for result in state.latest_results.values()
            for interrupt in result.interrupts
        ]
        return GraphResult(
            status=state.status,
            results=dict(state.latest_results),
            accumulated_metrics=state.accumulated_metrics,
            execution_count=state.execution_count,
            execution_time=state.execution_time,
            interrupts=interrupts,
            entry_points=sorted(self.entry_points),
            terminal_nodes=terminals,
            node_history={key: list(value) for key, value in state.node_history.items()},
            execution_order=list(state.execution_order),
            skipped_nodes=sorted(state.skipped_nodes),
            completed_node_ids=sorted(
                node_id
                for node_id, result in state.latest_results.items()
                if result.status is Status.COMPLETED
            ),
            failed_node_ids=sorted(state.failed_nodes),
            interrupted_node_ids=sorted(state.interrupted_nodes),
            error=state.error,
        )

    def _group_edges(
        self,
        attribute: str,
    ) -> dict[str, tuple[GraphEdge, ...]]:
        grouped: dict[str, list[GraphEdge]] = {}
        for edge in self.edges:
            key = getattr(edge, attribute)
            grouped.setdefault(key, []).append(edge)
        return {
            key: tuple(sorted(values, key=lambda edge: (edge.source, edge.target)))
            for key, values in grouped.items()
        }

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

    def serialize_state(self) -> dict[str, Any]:
        raise NotImplementedError(
            "Graph resume requires AgentResult serialization, planned for phase 5."
        )

    def deserialize_state(self, payload: dict[str, Any]) -> None:
        del payload
        raise NotImplementedError(
            "Graph resume requires AgentResult serialization, planned for phase 5."
        )

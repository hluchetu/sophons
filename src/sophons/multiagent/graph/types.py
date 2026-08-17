"""Types for directed multi-agent graphs."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, TypeGuard, cast

from sophons.agents import Agent, AgentMetrics
from sophons.multiagent.base import MultiAgentBase, MultiAgentResult, NodeResult, Status

JoinPolicy = Literal["all", "any"]
GraphExecutor = Agent | MultiAgentBase
InputMapper = Callable[["GraphState"], Any]


class EdgeConditionWithContext(Protocol):
    def __call__(
        self,
        state: GraphState,
        *,
        invocation_state: dict[str, Any],
        **kwargs: Any,
    ) -> bool: ...


LegacyEdgeCondition = Callable[["GraphState"], bool]
EdgeCondition = LegacyEdgeCondition | EdgeConditionWithContext


def accepts_invocation_state(
    condition: EdgeCondition,
) -> TypeGuard[EdgeConditionWithContext]:
    try:
        return "invocation_state" in inspect.signature(condition).parameters
    except (TypeError, ValueError):
        return False


@dataclass(frozen=True, eq=False)
class GraphNode:
    """Immutable configuration for one graph node."""

    node_id: str
    executor: GraphExecutor
    join: JoinPolicy = "all"
    input_mapper: InputMapper | None = None
    timeout: float | None = None

    def __hash__(self) -> int:
        return hash(self.node_id)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, GraphNode) and self.node_id == other.node_id


@dataclass(frozen=True, eq=False)
class GraphEdge:
    """Directed control-flow edge with an optional condition."""

    source: str
    target: str
    condition: EdgeCondition | None = field(default=None, compare=False, hash=False)
    label: str | None = field(default=None, compare=False, hash=False)

    def __hash__(self) -> int:
        return hash((self.source, self.target))

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, GraphEdge)
            and self.source == other.source
            and self.target == other.target
        )

    def should_traverse(
        self,
        state: GraphState,
        *,
        invocation_state: dict[str, Any],
    ) -> bool:
        if self.condition is None:
            return True
        if accepts_invocation_state(self.condition):
            contextual = cast(EdgeConditionWithContext, self.condition)
            return contextual(state, invocation_state=invocation_state)
        legacy = cast(LegacyEdgeCondition, self.condition)
        return legacy(state)


@dataclass(frozen=True)
class NodeExecution:
    """One immutable entry in a node's execution history."""

    node_id: str
    execution_index: int
    result: NodeResult


@dataclass
class GraphState:
    """Invocation-local state passed to conditions and input mappers."""

    task: str
    invocation_state: dict[str, Any] = field(default_factory=dict)
    status: Status = Status.PENDING
    latest_results: dict[str, NodeResult] = field(default_factory=dict)
    node_history: dict[str, list[NodeExecution]] = field(default_factory=dict)
    execution_order: list[str] = field(default_factory=list)
    skipped_nodes: set[str] = field(default_factory=set)
    failed_nodes: set[str] = field(default_factory=set)
    interrupted_nodes: set[str] = field(default_factory=set)
    edge_values: dict[tuple[str, str], bool] = field(default_factory=dict)
    consumed_signals: dict[str, dict[str, int]] = field(default_factory=dict)
    accumulated_metrics: AgentMetrics = field(default_factory=AgentMetrics)
    execution_count: int = 0
    execution_time: int = 0
    error: str | None = None

    def output_for(self, node_id: str) -> Any:
        try:
            return self.latest_results[node_id].output
        except KeyError as error:
            raise KeyError(f"Node {node_id!r} has not produced an output.") from error

    def executions_for(self, node_id: str) -> tuple[NodeExecution, ...]:
        return tuple(self.node_history.get(node_id, ()))

    def execution_count_for(self, node_id: str) -> int:
        return len(self.node_history.get(node_id, ()))


@dataclass
class GraphResult(MultiAgentResult):
    """Final graph result with deterministic terminal outputs and history."""

    entry_points: list[str] = field(default_factory=list)
    terminal_nodes: list[str] = field(default_factory=list)
    node_history: dict[str, list[NodeExecution]] = field(default_factory=dict)
    execution_order: list[str] = field(default_factory=list)
    skipped_nodes: list[str] = field(default_factory=list)
    completed_node_ids: list[str] = field(default_factory=list)
    failed_node_ids: list[str] = field(default_factory=list)
    interrupted_node_ids: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def outputs(self) -> dict[str, Any]:
        return {
            node_id: self.results[node_id].output
            for node_id in self.terminal_nodes
            if node_id in self.results
            and self.results[node_id].status is Status.COMPLETED
        }

    @property
    def output(self) -> Any:
        outputs = self.outputs
        if len(outputs) == 1:
            return next(iter(outputs.values()))
        return outputs or None

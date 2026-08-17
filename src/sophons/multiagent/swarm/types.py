"""Data types used by Sophons swarm orchestration."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from sophons.agents import Agent, AgentMetrics
from sophons.multiagent.base import MultiAgentResult, NodeResult, Status


@dataclass(frozen=True, slots=True)
class Handoff:
    """Structured request to transfer control to another swarm member."""

    target: str
    message: str
    reason: str | None = None
    context: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class HandoffRecord:
    """A committed handoff between two swarm members."""

    source: str
    target: str
    message: str
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class SwarmNode:
    """Configuration for one member of a swarm."""

    node_id: str
    agent: Agent
    description: str = ""
    allowed_handoffs: frozenset[str] = frozenset()


@dataclass
class SharedContext:
    """Bounded, public information shared between swarm members."""

    max_entries: int = 20
    entries: list[dict[str, Any]] = field(default_factory=list)

    def add(self, source: str, values: dict[str, Any]) -> None:
        try:
            json.dumps(values)
        except (TypeError, ValueError) as error:
            raise ValueError("Swarm context must be JSON serializable.") from error

        self.entries.append({"source": source, "values": values})
        if len(self.entries) > self.max_entries:
            del self.entries[: len(self.entries) - self.max_entries]


@dataclass
class SwarmState:
    """Invocation-local state for a swarm run."""

    task: str
    current_node_id: str
    status: Status = Status.PENDING
    results: dict[str, NodeResult] = field(default_factory=dict)
    node_history: list[str] = field(default_factory=list)
    handoff_history: list[HandoffRecord] = field(default_factory=list)
    shared_context: SharedContext = field(default_factory=SharedContext)
    accumulated_metrics: AgentMetrics = field(default_factory=AgentMetrics)
    execution_count: int = 0
    execution_time: int = 0
    stop_reason: str | None = None


@dataclass
class SwarmResult(MultiAgentResult):
    """Final result of a swarm invocation."""

    entry_agent: str = ""
    last_active_agent: str = ""
    node_history: list[str] = field(default_factory=list)
    handoff_history: list[HandoffRecord] = field(default_factory=list)
    shared_context: SharedContext = field(default_factory=SharedContext)
    iteration_count: int = 0
    handoff_count: int = 0
    stop_reason: str | None = None

    @property
    def output(self) -> Any:
        result = self.results.get(self.last_active_agent)
        return None if result is None else result.output

"""Shared foundations for Sophons multi-agent orchestration."""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from sophons.agents import AgentMetrics, AgentResult
from sophons.multiagent.interrupts import Interrupt


class Status(str, Enum):
    """Execution status shared by orchestrators and their nodes."""

    PENDING = "pending"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


@dataclass
class NodeResult:
    """Execution state and latest result for one multi-agent node."""

    result: AgentResult | MultiAgentResult | Exception | None = None
    execution_time: int = 0
    status: Status = Status.PENDING
    accumulated_metrics: AgentMetrics = field(default_factory=AgentMetrics)
    execution_count: int = 0
    interrupts: list[Interrupt] = field(default_factory=list)

    @property
    def success(self) -> bool:
        """Whether the node completed successfully."""

        return self.status is Status.COMPLETED

    @property
    def output(self) -> Any:
        """Return the node's structured output or text response."""

        if isinstance(self.result, AgentResult):
            return self.result.output if self.result.output is not None else self.result.message

        if isinstance(self.result, MultiAgentResult):
            return self.result.output

        return None

    def get_agent_results(self) -> list[AgentResult]:
        """Return all underlying agent results, including nested ones."""

        if isinstance(self.result, AgentResult):
            return [self.result]

        if isinstance(self.result, MultiAgentResult):
            return self.result.get_agent_results()

        return []

    def __str__(self) -> str:
        return "" if self.result is None else str(self.result)


@dataclass
class MultiAgentResult:
    """Result returned by a Graph, Swarm, or nested orchestrator."""

    status: Status = Status.PENDING
    results: dict[str, NodeResult] = field(default_factory=dict)
    accumulated_metrics: AgentMetrics = field(default_factory=AgentMetrics)
    execution_count: int = 0
    execution_time: int = 0
    interrupts: list[Interrupt] = field(default_factory=list)

    @property
    def success(self) -> bool:
        """Whether the complete orchestration finished successfully."""

        return self.status is Status.COMPLETED

    @property
    def output(self) -> Any:
        """Return the output of the most recently completed node."""

        for node_result in reversed(self.results.values()):
            if node_result.status is Status.COMPLETED:
                return node_result.output

        return None

    def get_agent_results(self) -> list[AgentResult]:
        """Return all underlying agent results, flattening nested systems."""

        agent_results: list[AgentResult] = []

        for node_result in self.results.values():
            agent_results.extend(node_result.get_agent_results())

        return agent_results

    def __str__(self) -> str:
        output = self.output
        return "" if output is None else str(output)


class MultiAgentBase(ABC):
    """Base class for Graph, Swarm, and future orchestration patterns."""

    def __init__(self, *, id: str) -> None:
        if not id.strip():
            raise ValueError("Multi-agent id cannot be empty.")

        self.id = id
        self._invocation_start_time: float | None = None

    def _start_invocation(self) -> None:
        """Start measuring an active execution interval."""

        self._invocation_start_time = time.perf_counter()

    def _execution_time_with_active_interval(self, committed_ms: int = 0) -> int:
        """Return committed time plus the currently active interval."""

        if self._invocation_start_time is None:
            return committed_ms

        active_ms = (time.perf_counter() - self._invocation_start_time) * 1000
        return committed_ms + round(active_ms)

    def _commit_active_interval(self, committed_ms: int = 0) -> int:
        """Commit the active interval and stop its timer."""

        execution_time = self._execution_time_with_active_interval(committed_ms)
        self._invocation_start_time = None
        return execution_time

    @abstractmethod
    async def run(
        self,
        input: str,
        *,
        invocation_state: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> MultiAgentResult:
        """Run the multi-agent system asynchronously."""

        raise NotImplementedError

    async def stream(
        self,
        input: str,
        *,
        invocation_state: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[dict[str, Any]]:
        """Run the system and yield its final result."""

        result = await self.run(
            input,
            invocation_state=invocation_state,
            **kwargs,
        )
        yield {"result": result}

    def run_sync(
        self,
        input: str,
        *,
        invocation_state: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> MultiAgentResult:
        """Run the multi-agent system from synchronous code."""

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise RuntimeError(
                "MultiAgentBase.run_sync() cannot be called from inside "
                "a running event loop. Use 'await run()' instead."
            )

        return asyncio.run(
            self.run(
                input,
                invocation_state=invocation_state,
                **kwargs,
            )
        )

    def __call__(
        self,
        input: str,
        *,
        invocation_state: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> MultiAgentResult:
        """Run the multi-agent system synchronously."""

        return self.run_sync(
            input,
            invocation_state=invocation_state,
            **kwargs,
        )

    def serialize_state(self) -> dict[str, Any]:
        """Serialize execution state in a concrete orchestrator."""

        raise NotImplementedError(
            f"{type(self).__name__} does not implement state serialization."
        )

    def deserialize_state(self, payload: dict[str, Any]) -> None:
        """Restore execution state in a concrete orchestrator."""

        raise NotImplementedError(
            f"{type(self).__name__} does not implement state deserialization."
        )

    def add_hook(self, callback: Callable[..., Any]) -> None:
        """Register a lifecycle hook when supported by an orchestrator."""

        raise NotImplementedError(
            f"{type(self).__name__} does not support multi-agent hooks."
        )

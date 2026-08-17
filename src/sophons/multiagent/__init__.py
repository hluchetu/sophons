"""Multi-agent orchestration patterns for Sophons."""

from __future__ import annotations

from sophons.multiagent.base import (
    MultiAgentBase,
    MultiAgentResult,
    NodeResult,
    Status,
)
from sophons.multiagent.graph import (
    EdgeCondition,
    EdgeConditionWithContext,
    Graph,
    GraphBuilder,
    GraphEdge,
    GraphNode,
    GraphResult,
    GraphState,
    LegacyEdgeCondition,
)
from sophons.multiagent.interrupts import Interrupt
from sophons.multiagent.swarm import (
    Handoff,
    HandoffRecord,
    HandoffTool,
    SharedContext,
    Swarm,
    SwarmNode,
    SwarmResult,
    SwarmState,
)

__all__ = [
    "EdgeCondition",
    "EdgeConditionWithContext",
    "Graph",
    "GraphBuilder",
    "GraphEdge",
    "GraphNode",
    "GraphResult",
    "GraphState",
    "Handoff",
    "HandoffRecord",
    "HandoffTool",
    "Interrupt",
    "LegacyEdgeCondition",
    "MultiAgentBase",
    "MultiAgentResult",
    "NodeResult",
    "Status",
    "SharedContext",
    "Swarm",
    "SwarmNode",
    "SwarmResult",
    "SwarmState",
]

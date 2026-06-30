from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal

# -----stop Reason---------------------------------------------------------------------


class StopReason(str, Enum):
    END_TURN = "end_turn"  # model decided it is done
    MAX_STEPS = "max_steps"  # the loop hit the step limit
    MAX_TOKENS = "max_tokens"  # token budget exhausted
    MAX_RUNTIME = "max_runtime"  # time limit hit
    CANCELLED = "cancelled"  # cancelled from outside
    ERROR = "error"  # unrecoverable failure


# -----Tool Types--------------------------------------------------------------------------


@dataclass
class ToolUse:
    """The model's request to call a tool"""

    tool_use_id: str
    name: str
    input: dict[str, Any]


@dataclass
class ToolResult:
    """The result of executing a tool"""

    tool_use_id: str
    status: Literal["success", "error"]
    content: str


# ------metrics
@dataclass
class AgentMetrics:
    steps: int = 0
    model_calls: int = 0
    tool_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    duration_ms: float = 0


@dataclass
class AgentResult:
    """
    The full result of an agent run — mirrors Strands AgentResult.
    Returned by agent.run() to the developer.
    """

    stop_reason: StopReason  # why the agent stopped
    message: str  # the final answer text
    metrics: AgentMetrics  # performance numbers
    tool_uses: list[ToolUse]  # every tool the model requested
    tool_results: list[ToolResult]  # every tool result received
    success: bool  # did it reach end_turn cleanly?
    error: str | None = None  # error message if stop_reason is ERROR

    def __str__(self) -> str:
        return self.message

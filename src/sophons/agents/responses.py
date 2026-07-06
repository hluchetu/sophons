from __future__ import annotations

from dataclasses import dataclass, field
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
class ToolStats:
    """Per-tool call counts and latency for a single agent run."""

    calls: int = 0
    errors: int = 0
    total_ms: float = 0.0

    @property
    def avg_ms(self) -> float:
        return self.total_ms / self.calls if self.calls else 0.0


@dataclass
class AgentMetrics:
    steps: int = 0
    model_calls: int = 0
    tool_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0  # tokens read from provider cache
    cache_write_tokens: int = 0  # tokens written to provider cache
    duration_ms: float = 0
    per_tool: dict[str, ToolStats] = field(default_factory=dict)


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

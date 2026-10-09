from __future__ import annotations

import time
import math
from dataclasses import dataclass, field

from sophons.agents.responses import ToolStats

# ── RunLimits ──────────────────────────────────────────────────────────────────


@dataclass
class RunLimits:
    """
    Boundaries for a single agent run.
    Set once before the run starts. Never changes during the run.

    Example:
        limits = RunLimits(max_steps=5, max_runtime_seconds=60.0)
    """

    max_steps: int = 10
    # Counts every admitted main-model attempt, including failures and retries.
    max_model_calls: int = 20
    # Counts admitted dispatch attempts, including structured-response calls.
    # Denied/failed admitted calls consume a slot; skipped calls do not.
    max_tool_calls: int = 20
    max_tokens: int | None = None
    max_runtime_seconds: float = 300.0

    def __post_init__(self) -> None:
        if type(self.max_model_calls) is not int or self.max_model_calls < 0:
            raise ValueError("max_model_calls must be a nonnegative integer")
        if isinstance(self.max_runtime_seconds, bool) or not isinstance(self.max_runtime_seconds, (int, float)) or not math.isfinite(self.max_runtime_seconds) or self.max_runtime_seconds < 0:
            raise ValueError("max_runtime_seconds must be finite and nonnegative")
        if type(self.max_tool_calls) is not int or self.max_tool_calls < 0:
            raise ValueError("max_tool_calls must be a nonnegative integer")


# ── RunState ───────────────────────────────────────────────────────────────────


@dataclass
class RunState:
    """
    Live tracker for a single agent run.
    Created fresh at the start of every agent.run() call.
    Updated by the loop at every step.

    Example:
        state = RunState()
        state.step_count += 1
        state.elapsed_seconds()
    """

    step_count: int = 0
    model_call_count: int = 0
    tool_call_count: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    per_tool: dict[str, ToolStats] = field(default_factory=dict)
    started_at: float = field(default_factory=time.monotonic)

    def reserve_model_call(self, limits: RunLimits) -> bool:
        """Reserve before invoking the model, including each retry attempt."""
        if self.model_call_count >= limits.max_model_calls:
            return False
        self.model_call_count += 1
        return True

    def reserve_tool_call(self, limits: RunLimits) -> bool:
        """Consume one slot before dispatch; never exceed the configured count."""
        if self.tool_call_count >= limits.max_tool_calls:
            return False
        self.tool_call_count += 1
        return True

    def record_tool_call(self, tool_name: str, duration_ms: float, *, error: bool) -> None:
        """Update per-tool stats after a tool finishes executing."""
        stats = self.per_tool.setdefault(tool_name, ToolStats())
        stats.calls += 1
        stats.total_ms += duration_ms
        if error:
            stats.errors += 1

    def elapsed_seconds(self) -> float:
        """How many seconds have passed since this run started."""
        return time.monotonic() - self.started_at

    def total_tokens(self) -> int:
        """Total tokens used so far in this run."""
        return self.input_tokens + self.output_tokens

    def exceeds(self, limits: RunLimits) -> str | None:
        """
        Check if any limit has been exceeded.
        Returns the exceeded limit name or None if all good.

        The loop calls this at the start of every iteration.

        Example:
            reason = state.exceeds(limits)
            if reason:
                stop(reason)
        """
        if self.step_count >= limits.max_steps:
            return "max_steps"
        if self.model_call_count >= limits.max_model_calls:
            return "max_model_calls"
        if self.tool_call_count >= limits.max_tool_calls:
            return "max_tool_calls"
        if self.elapsed_seconds() > limits.max_runtime_seconds:
            return "max_runtime"
        if limits.max_tokens and self.total_tokens() >= limits.max_tokens:
            return "max_tokens"
        return None

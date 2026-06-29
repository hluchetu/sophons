from __future__ import annotations

import time
from dataclasses import dataclass, field

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
    max_model_calls: int = 20
    max_tool_calls: int = 20
    max_tokens: int | None = None
    max_runtime_seconds: float = 300.0


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
    started_at: float = field(default_factory=time.monotonic)

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

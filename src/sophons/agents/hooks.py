from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Type

from sophons.agents.responses import AgentResult, ToolResult, ToolUse
from sophons.models.messages import Message

# ── Events ─────────────────────────────────────────────────────────────────────


@dataclass
class AgentStarted:
    """Fired when agent.run() is called."""

    input: str
    session_id: str | None = None


@dataclass
class AgentFinished:
    """Fired when agent.run() completes successfully."""

    result: AgentResult
    session_id: str | None = None


@dataclass
class AgentFailed:
    """Fired when agent.run() fails with an unrecoverable error."""

    error: Exception
    step: int
    session_id: str | None = None


@dataclass
class BeforeModelCall:
    """Fired just before the model is called."""

    messages: list[Message]
    step: int


@dataclass
class AfterModelCall:
    """Fired after the model returns a response."""

    message: Message
    step: int
    duration_ms: float


@dataclass
class BeforeToolCall:
    """Fired just before a tool is executed."""

    tool_use: ToolUse
    step: int


@dataclass
class AfterToolCall:
    """Fired after a tool finishes executing."""

    tool_use: ToolUse
    tool_result: ToolResult
    step: int
    duration_ms: float


@dataclass
class MessageAdded:
    """Fired every time a message is added to the conversation."""

    message: Message
    step: int


# All event types in one place — useful for type hints
AgentHookEvent = (
    AgentStarted
    | AgentFinished
    | AgentFailed
    | BeforeModelCall
    | AfterModelCall
    | BeforeToolCall
    | AfterToolCall
    | MessageAdded
)


# ── HookRegistry ───────────────────────────────────────────────────────────────


class HookRegistry:
    """
    Registry of lifecycle hooks for the agent loop.

    The agent loop fires events at key checkpoints.
    Any code that registers a hook gets called when that event fires.

    Usage:
        hooks = HookRegistry()

        # register a hook
        hooks.register(AfterModelCall, lambda e: print(f"step {e.step}"))

        # the loop fires events
        hooks.invoke(AfterModelCall(message=msg, step=1, duration_ms=120.0))

        # check if anyone is listening
        hooks.has_hooks(AfterModelCall)

        # remove hooks for an event
        hooks.deregister(AfterModelCall)
    """

    def __init__(self) -> None:
        self._hooks: dict[Type, list[Callable]] = {}

    def register(
        self,
        event_type: Type | list[Type],
        callback: Callable,
    ) -> None:
        """
        Register a callback for one or more event types.

        Args:
            event_type: A single event type or a list of event types.
            callback:   A callable that accepts the event as its only argument.

        Example:
            hooks.register(AfterModelCall, my_logger)
            hooks.register([BeforeToolCall, AfterToolCall], my_tracer)
        """
        types = event_type if isinstance(event_type, list) else [event_type]
        for t in types:
            if t not in self._hooks:
                self._hooks[t] = []
            self._hooks[t].append(callback)

    def invoke(self, event: Any) -> None:
        """
        Invoke all callbacks registered for this event type.

        Args:
            event: The event instance to pass to each callback.

        Example:
            hooks.invoke(AgentStarted(input="hello"))
        """
        callbacks = self._hooks.get(type(event), [])
        for callback in callbacks:
            callback(event)

    def has_hooks(self, event_type: Type | None = None) -> bool:
        """
        Check if any callbacks are registered.

        Args:
            event_type: Check for a specific type, or None to check all.

        Returns:
            True if callbacks exist, False otherwise.

        Example:
            hooks.has_hooks(AfterModelCall)  # True or False
            hooks.has_hooks()                # any hooks at all?
        """
        if event_type is None:
            return any(len(cbs) > 0 for cbs in self._hooks.values())
        return bool(self._hooks.get(event_type))

    def deregister(self, event_type: Type | None = None) -> None:
        """
        Remove callbacks for one event type or all event types.

        Args:
            event_type: The type to clear, or None to clear everything.

        Example:
            hooks.deregister(AfterModelCall)  # remove only these
            hooks.deregister()                # remove everything
        """
        if event_type is None:
            self._hooks.clear()
        else:
            self._hooks.pop(event_type, None)

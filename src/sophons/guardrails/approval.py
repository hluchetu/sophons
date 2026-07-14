from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from sophons.guardrails.base import Boundary, GuardrailContext, GuardrailDecision


@dataclass(frozen=True, slots=True)
class ApprovalRequest:
    """Everything a human needs to decide one pending action."""

    boundary: Boundary
    reason: str  # why the guardrail wants a human
    value: Any  # the pending action (tool args at the tool boundary)
    tool_name: str | None = None
    session_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ApprovalDecision:
    """A human (or their delegate) said yes or no."""

    approved: bool
    approver: str  # who decided — lands in the trace
    note: str = ""  # optional comment, shown to the model on denial


@runtime_checkable
class Approver(Protocol):
    """Anything that can turn a pending action into a verdict."""

    async def approve(self, request: ApprovalRequest) -> ApprovalDecision: ...


class ConsoleApprover:
    """Ask the person at the terminal — the CLI-agent pattern.

    Runs input() in a worker thread so the event loop is not blocked
    while the human thinks.
    """

    def __init__(self, *, name: str = "console") -> None:
        self.name = name

    async def approve(self, request: ApprovalRequest) -> ApprovalDecision:
        tool = f" -> {request.tool_name}({request.value})" if request.tool_name else ""
        prompt = f"\n[approval needed] {request.reason}{tool}\nApprove? [y/N] "
        answer = await asyncio.to_thread(input, prompt)
        approved = answer.strip().lower() in ("y", "yes")
        return ApprovalDecision(approved=approved, approver=self.name)


class CallbackApprover:
    """Wrap an app-supplied async callable — the seam a service uses to
    route the request to its own approval UI, and what tests use to
    script verdicts deterministically."""

    def __init__(
        self,
        callback: Callable[[ApprovalRequest], Awaitable[ApprovalDecision]],
        *,
        name: str = "callback",
    ) -> None:
        self._callback = callback
        self.name = name

    async def approve(self, request: ApprovalRequest) -> ApprovalDecision:
        return await self._callback(request)


ApprovalRule = Callable[[dict[str, Any]], str | None]
"""Per-tool predicate: return a reason requiring approval, None to auto-allow."""


class ApprovalGuardrail:
    """
    Decides which tool calls need a human — the confirm counterpart of
    ToolPermissionGuardrail. A rule returning a reason sends the call to
    the configured approver; returning None lets it run unattended.

    Usage::

        ApprovalGuardrail(rules={
            "refund_order": lambda args: (
                f"amount {args.get('amount')} exceeds the auto-approve limit"
                if args.get("amount", 0) > 100 else None
            ),
        })
    """

    name = "approval"

    def __init__(
        self,
        *,
        rules: dict[str, ApprovalRule],
        message: str | None = None,
    ) -> None:
        self.rules = rules
        self.message = message

    async def check(
        self, value: Any, *, context: GuardrailContext
    ) -> GuardrailDecision:
        if context.boundary != "tool" or context.tool_name not in self.rules:
            return GuardrailDecision.allow()
        args = value if isinstance(value, dict) else {}
        reason = self.rules[context.tool_name](args)
        if reason:
            return GuardrailDecision.confirm(reason, message=self.message)
        return GuardrailDecision.allow()

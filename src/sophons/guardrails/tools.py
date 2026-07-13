from __future__ import annotations

from collections.abc import Callable
from typing import Any

from sophons.guardrails.base import GuardrailContext, GuardrailDecision

ArgumentRule = Callable[[dict[str, Any]], str | None]
"""Per-tool argument check: return a violation reason to block, None to allow."""


class ToolPermissionGuardrail:
    """
    Policy-based tool authorization: which tools may run, with what arguments.

    Args:
        allowed:        If given, every tool NOT in this set is blocked —
                        a strict allowlist for high-stakes agents.
        denied:         Tools that are always blocked, regardless of allowed.
        argument_rules: Per-tool argument checks, e.g. capping a refund:
                        ``{"refund_order": lambda args: "amount over limit"
                        if args.get("amount", 0) > 100 else None}``
        message:        User-facing text when a call is blocked.

    Only acts at the tool boundary; allows everything elsewhere.
    """

    name = "tool-permission"

    def __init__(
        self,
        *,
        allowed: set[str] | None = None,
        denied: set[str] | None = None,
        argument_rules: dict[str, ArgumentRule] | None = None,
        message: str | None = None,
    ) -> None:
        self.allowed = allowed
        self.denied = denied or set()
        self.argument_rules = argument_rules or {}
        self.message = message

    async def check(
        self, value: Any, *, context: GuardrailContext
    ) -> GuardrailDecision:
        if context.boundary != "tool" or context.tool_name is None:
            return GuardrailDecision.allow()
        tool = context.tool_name

        if tool in self.denied:
            return GuardrailDecision.block(
                f"tool {tool!r} is denied by policy", message=self.message
            )
        if self.allowed is not None and tool not in self.allowed:
            return GuardrailDecision.block(
                f"tool {tool!r} is not on the allowlist", message=self.message
            )

        rule = self.argument_rules.get(tool)
        if rule is not None:
            args = value if isinstance(value, dict) else {}
            violation = rule(args)
            if violation:
                return GuardrailDecision.block(
                    f"{tool} arguments rejected: {violation}",
                    message=self.message,
                )

        return GuardrailDecision.allow()

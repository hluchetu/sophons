"""Structured handoff tool used by a Sophons swarm."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from sophons.multiagent.swarm.types import Handoff
from sophons.tools.base import ToolArgs, ToolResult, ToolSchema


@dataclass(frozen=True, slots=True)
class HandoffTool:
    """Agent tool that records a structured handoff request."""

    request_handoff: Callable[[Handoff], None]
    name: str = "handoff_to_agent"
    description: str = (
        "Transfer control to another swarm specialist. Use this only when "
        "another agent is better suited to continue the task."
    )

    @property
    def args_schema(self) -> ToolSchema:
        return {
            "type": "object",
            "properties": {
                "agent_name": {"type": "string"},
                "message": {"type": "string"},
                "reason": {"type": "string"},
                "context": {"type": "object"},
            },
            "required": ["agent_name", "message"],
        }

    def call(self, args: ToolArgs) -> ToolResult:
        target = args.get("agent_name")
        message = args.get("message")
        if not isinstance(target, str) or not target.strip():
            return {"status": "error", "error": "agent_name must be non-empty"}
        if not isinstance(message, str) or not message.strip():
            return {"status": "error", "error": "message must be non-empty"}

        context = args.get("context", {})
        if not isinstance(context, dict):
            return {"status": "error", "error": "context must be an object"}

        reason = args.get("reason")
        request = Handoff(
            target=target,
            message=message,
            reason=reason if isinstance(reason, str) else None,
            context=context,
        )
        try:
            self.request_handoff(request)
        except ValueError as error:
            return {"status": "error", "error": str(error)}

        return {
            "status": "success",
            "target": target,
            "message": message,
        }

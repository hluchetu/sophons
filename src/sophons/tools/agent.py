from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sophons.tools.base import ToolArgs, ToolResult, ToolSchema


@dataclass(frozen=True, slots=True)
class AgentTool:
    """A tool that wraps a Sophons Agent — lets a supervisor delegate to a specialist."""

    name: str
    description: str
    agent: Any

    @property
    def args_schema(self) -> ToolSchema:
        return {
            "type": "object",
            "properties": {"input": {"type": "string"}},
            "required": ["input"],
        }

    def call(self, args: ToolArgs) -> ToolResult:
        result = self.agent(args["input"])
        return {"result": result.message}

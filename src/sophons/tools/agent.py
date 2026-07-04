from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from sophons.tools.base import ToolArgs, ToolResult, ToolSchema


@runtime_checkable
class Runnable(Protocol):
    """Anything with an async run() that returns a result with a .message attribute."""

    async def run(self, input: str) -> Any: ...


@dataclass(frozen=True, slots=True)
class AgentTool:
    """A tool that wraps a Sophons Agent — lets a supervisor delegate to a specialist."""

    name: str
    description: str
    agent: Runnable

    @property
    def args_schema(self) -> ToolSchema:
        return {
            "type": "object",
            "properties": {"input": {"type": "string"}},
            "required": ["input"],
        }

    async def call(self, args: ToolArgs) -> ToolResult:
        result = await self.agent.run(args["input"])
        return {"result": result.message}

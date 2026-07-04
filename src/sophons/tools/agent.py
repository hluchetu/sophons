from __future__ import annotations

from typing import Any

from sophons.tools.decorator import FunctionTool


def as_tool(agent: Any, *, name: str, description: str) -> FunctionTool:
    """Wrap a Sophons Agent as a FunctionTool so a supervisor can call it like any other tool."""

    def run(input: str) -> str:
        return agent(input).message

    return FunctionTool(
        name=name,
        description=description,
        args_schema={
            "type": "object",
            "properties": {"input": {"type": "string"}},
            "required": ["input"],
        },
        fn=run,
    )

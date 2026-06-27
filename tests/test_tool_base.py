from __future__ import annotations

import pytest

from sophons.tools import AsyncTool, Tool


class EchoTool:
    name = "echo"
    description = "Echo a value."
    args_schema = {
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
    }

    def call(self, args: dict) -> dict:
        return {"ok": True, "value": args["value"]}


class AsyncEchoTool:
    name = "echo"
    description = "Echo a value."
    args_schema = {
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
    }

    async def call(self, args: dict) -> dict:
        return {"ok": True, "value": args["value"]}


def test_tool_protocol_accepts_matching_object() -> None:
    tool = EchoTool()

    assert isinstance(tool, Tool)
    assert tool.call({"value": "hello"}) == {"ok": True, "value": "hello"}


@pytest.mark.asyncio
async def test_async_tool_protocol_accepts_matching_object() -> None:
    tool = AsyncEchoTool()

    assert isinstance(tool, AsyncTool)
    assert await tool.call({"value": "hello"}) == {"ok": True, "value": "hello"}

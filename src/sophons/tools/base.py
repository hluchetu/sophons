from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

ToolArgs = dict[str, Any]
ToolResult = dict[str, Any]
ToolSchema = dict[str, Any]


@runtime_checkable
class Tool(Protocol):
    """Sync tool contract: structured arguments in, structured result out."""

    @property
    def name(self) -> str: ...

    @property
    def description(self) -> str: ...

    @property
    def args_schema(self) -> ToolSchema: ...

    def call(self, args: ToolArgs) -> ToolResult: ...


@runtime_checkable
class AsyncTool(Protocol):
    """Async tool contract: structured arguments in, structured result out."""

    @property
    def name(self) -> str: ...

    @property
    def description(self) -> str: ...

    @property
    def args_schema(self) -> ToolSchema: ...

    async def call(self, args: ToolArgs) -> ToolResult: ...

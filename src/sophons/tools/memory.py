from __future__ import annotations

from typing import Any

from sophons.memory import MemoryManager
from sophons.memory.long_term.entry import ALL_MEMORY_TYPES, MemoryEntry
from sophons.tools.base import ToolArgs, ToolResult, ToolSchema


class SearchMemoryTool:
    """Agent-callable tool for searching long-term memory."""

    name = "search_memory"
    description = "Search long-term memory for information relevant to a query."
    args_schema: ToolSchema = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "limit": {"type": "integer", "default": 5},
        },
        "required": ["query"],
    }

    def __init__(
        self,
        memory_manager: MemoryManager,
        namespace: tuple[str, ...],
    ) -> None:
        self._memory_manager = memory_manager
        self._namespace = namespace

    async def call(self, args: ToolArgs) -> ToolResult:
        query = str(args.get("query", "")).strip()
        limit = int(args.get("limit") or 5)
        if not query:
            return {"memories": [], "text": ""}

        entries = await self._memory_manager.search(
            query=query,
            namespace=self._namespace,
            limit=limit,
        )
        return {
            "memories": [entry.to_dict() for entry in entries],
            "text": "\n".join(
                f"- [{entry.memory_type}] {entry.key}: {entry.content}"
                for entry in entries
            ),
        }


class AddMemoryTool:
    """Agent-callable tool for writing an explicit long-term memory."""

    name = "add_memory"
    description = "Add a durable long-term memory for future interactions."
    args_schema: ToolSchema = {
        "type": "object",
        "properties": {
            "content": {"type": "string"},
            "memory_type": {
                "type": "string",
                "enum": list(ALL_MEMORY_TYPES),
                "default": "semantic",
            },
            "key": {"type": "string"},
            "importance": {"type": "number"},
            "metadata": {"type": "object"},
        },
        "required": ["content", "key"],
    }

    def __init__(
        self,
        memory_manager: MemoryManager,
        namespace: tuple[str, ...],
    ) -> None:
        self._memory_manager = memory_manager
        self._namespace = namespace

    async def call(self, args: ToolArgs) -> ToolResult:
        content = str(args.get("content", "")).strip()
        key = str(args.get("key", "")).strip()
        if not content:
            return {"error": "content is required"}
        if not key:
            return {"error": "key is required"}
        raw_type = str(args.get("memory_type") or "semantic").strip()
        if raw_type not in ALL_MEMORY_TYPES:
            raw_type = "semantic"
        importance = _optional_float(args.get("importance"))
        metadata = args.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}

        entry = MemoryEntry(
            memory_type=raw_type,  # type: ignore[arg-type]
            namespace=self._namespace,
            key=key,
            content=content,
            importance=importance,
            metadata=dict(metadata),
        )
        stored = await self._memory_manager.remember(entry)
        return {"memory": stored.to_dict()}


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

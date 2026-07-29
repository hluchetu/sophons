from __future__ import annotations

import pytest

from sophons.memory import (
    InMemoryStorage,
    LexicalRetriever,
    MemoryManager,
    MemoryStore,
    MemoryStoreConfig,
)
from sophons.tools.memory import AddMemoryTool, SearchMemoryTool


@pytest.mark.asyncio
async def test_memory_tools_add_and_search_explicit_memories() -> None:
    store = MemoryStore(storage=InMemoryStorage(), retrievers=[LexicalRetriever()])
    memory = MemoryManager(
        stores=[MemoryStoreConfig(name="main", description="main", store=store)]
    )
    namespace = ("user", "alice")

    add = AddMemoryTool(memory, namespace)
    search = SearchMemoryTool(memory, namespace)

    added = await add.call(
        {
            "key": "style",
            "content": "User likes direct technical notes.",
            "memory_type": "preference",
            "importance": 0.9,
        }
    )
    found = await search.call({"query": "technical notes", "limit": 3})

    assert added["memory"]["key"] == "style"
    assert found["memories"][0]["content"] == "User likes direct technical notes."
    assert "[preference] style" in found["text"]

from __future__ import annotations

import pytest

from sophons.agents import Agent, MemoryConfig
from sophons.memory import (
    InMemoryStorage,
    MemoryEntry,
    MemoryManager,
    MemoryStore,
    MemoryStoreConfig,
)
from sophons.models import Message


class CapturingModel:
    def __init__(self) -> None:
        self.calls: list[list[Message]] = []

    def invoke(self, messages: list[Message], tools=None) -> Message:
        self.calls.append(messages)
        return Message(role="assistant", content="done")


@pytest.mark.asyncio
async def test_agent_injects_relevant_memory_into_current_turn() -> None:
    store = MemoryStore(storage=InMemoryStorage())
    memory = MemoryManager(
        stores=[MemoryStoreConfig(name="main", description="main", store=store)]
    )
    await memory.remember(
        MemoryEntry(
            memory_type="preference",
            namespace=("user", "alice"),
            key="style",
            content="User prefers first-principles explanations.",
            importance=1.0,
        )
    )
    model = CapturingModel()

    agent = Agent(
        model=model,
        memory_manager=memory,
        memory_config=MemoryConfig(
            namespace=("user", "alice"),
            extract_after_run=False,
        ),
    )

    result = await agent.run("Explain memory.")

    assert result.message == "done"
    user_message = model.calls[0][-1]
    assert user_message.role == "user"
    assert "Relevant long-term memory:" in user_message.content
    assert "User prefers first-principles explanations." in user_message.content
    assert "Current user message:\nExplain memory." in user_message.content


@pytest.mark.asyncio
async def test_agent_uses_memory_manager_defaults_when_config_is_omitted() -> None:
    store = MemoryStore(storage=InMemoryStorage())
    memory = MemoryManager(
        stores=[MemoryStoreConfig(name="main", description="main", store=store)],
        namespace=("user", "alice"),
        extract_after_run=False,
        context_header="Remembered learning context:",
    )
    await memory.remember(
        MemoryEntry(
            memory_type="preference",
            namespace=("user", "alice"),
            key="style",
            content="User likes examples connected to local code.",
        )
    )
    model = CapturingModel()

    agent = Agent(model=model, memory_manager=memory)

    await agent.run("Explain the agent memory change.")

    user_message = model.calls[0][-1]
    assert "Remembered learning context:" in user_message.content
    assert "User likes examples connected to local code." in user_message.content

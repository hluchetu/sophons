from __future__ import annotations

import pytest

from sophons.memory import LLMMemoryExtractor, MemoryExtractionRequest
from sophons.models import Message


class SyncExtractionModel:
    def invoke(self, messages: list[Message], tools=None) -> Message:
        return Message(
            role="assistant",
            content=(
                '{"records":[{"action":"create","memory_type":"preference",'
                '"key":"user.style","content":"User prefers concise answers.",'
                '"importance":0.8,"metadata":{}}]}'
            ),
        )


@pytest.mark.asyncio
async def test_llm_memory_extractor_accepts_sync_chat_models() -> None:
    extractor = LLMMemoryExtractor(model=SyncExtractionModel())

    result = await extractor.extract(
        MemoryExtractionRequest(
            namespace=("user", "alice"),
            messages=[Message(role="user", content="I prefer concise answers.")],
        )
    )

    assert len(result.entries) == 1
    assert result.entries[0].key == "user.style"
    assert result.entries[0].content == "User prefers concise answers."

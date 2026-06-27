from __future__ import annotations

from sophons import Message


def test_message_defaults() -> None:
    message = Message(role="user", content="hello")

    assert message.role == "user"
    assert message.content == "hello"
    assert message.metadata == {}
    assert message.id is None


def test_message_copy_helpers() -> None:
    message = Message(role="assistant", content="hello", metadata={"model": "test"})

    updated = message.with_metadata(latency_ms=10).with_id("msg_1")

    assert message.metadata == {"model": "test"}
    assert message.id is None
    assert updated.metadata == {"model": "test", "latency_ms": 10}
    assert updated.id == "msg_1"


def test_message_round_trip_dict() -> None:
    message = Message(
        id="msg_1",
        role="tool",
        content='{"ok": true}',
        metadata={"tool_name": "search_memory"},
    )

    assert Message.from_dict(message.to_dict()) == message

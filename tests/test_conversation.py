"""Context management: what each ConversationManager actually keeps.

These managers had no test coverage, which is how a SummarizingManager that
called a non-existent .chat() method survived. Several tests below pin down
current behaviour rather than ideal behaviour — they say so where that is
the case, so a future fix has to change them deliberately.
"""

from __future__ import annotations

import pytest

from sophons.agents.conversation import (
    ContextBudgetExceededError,
    SlidingWindowManager,
    SummarizingManager,
    TokenBudgetManager,
    ToolInteractionCompactor,
)
from sophons.models import Message


class CharTokenCounter:
    """Roughly four characters per token — deterministic, dependency-free."""

    def count_message(self, message: Message) -> int:
        return max(1, len(message.content) // 4)


class CountingModel:
    """A ChatModel that records how many times it was asked to summarize."""

    def __init__(self) -> None:
        self.calls = 0

    def invoke(self, messages: list[Message], tools: list | None = None) -> Message:
        self.calls += 1
        return Message(role="assistant", content="SUMMARY")


def msg(role: str, content: str, id: str, **metadata) -> Message:
    return Message(role=role, content=content, id=id, metadata=metadata or {})


def tool_exchange(n: int) -> list[Message]:
    """An assistant tool call plus its result — one atomic interaction."""
    return [
        msg(
            "assistant",
            "",
            f"a{n}",
            tool_calls=[
                {
                    "tool_use_id": f"c{n}",
                    "name": "branch_hours",
                    "input": {"branch": "x"},
                }
            ],
        ),
        msg("tool", f"result {n}", f"t{n}", tool_use_id=f"c{n}", name="branch_hours"),
    ]


# ---------------------------------------------------------------------------
# SlidingWindowManager
# ---------------------------------------------------------------------------


def test_sliding_window_keeps_last_n_plus_system():
    history = [
        msg("system", "sys", "s"),
        *[msg("user", f"turn {i}", f"m{i}") for i in range(5)],
    ]
    kept = SlidingWindowManager(max_messages=2).prepare(history)

    # max_messages counts non-system messages only, so 2 + system = 3.
    assert [m.id for m in kept] == ["s", "m3", "m4"]


def test_sliding_window_hoists_a_late_system_message():
    """Current behaviour: system messages move to the front, not kept in place."""
    history = [
        msg("user", "first", "m0"),
        msg("system", "injected later", "s"),
        msg("user", "second", "m1"),
    ]
    kept = SlidingWindowManager(max_messages=2).prepare(history)

    assert kept[0].id == "s"  # reordered, despite arriving second


def test_sliding_window_can_split_a_tool_pair():
    """Current behaviour: an orphaned tool result can be sent without its call.

    Some providers reject that outright. TokenBudgetManager keeps such pairs
    whole; the sliding window makes no such promise.
    """
    history = [msg("user", "q", "m0"), *tool_exchange(1)]
    kept = SlidingWindowManager(max_messages=1).prepare(history)

    assert [m.id for m in kept] == ["t1"]
    assert kept[0].role == "tool"  # the matching assistant call is gone


def test_sliding_window_rejects_zero():
    with pytest.raises(ValueError):
        SlidingWindowManager(max_messages=0)


# ---------------------------------------------------------------------------
# TokenBudgetManager
# ---------------------------------------------------------------------------


def test_token_budget_keeps_tool_pairs_whole():
    history = [msg("user", "q" * 40, "m0"), *tool_exchange(1)]
    kept = TokenBudgetManager(max_tokens=8, token_counter=CharTokenCounter()).prepare(
        history
    )
    ids = [m.id for m in kept]

    # Either both halves of the interaction survive, or neither does.
    assert ("a1" in ids) == ("t1" in ids)


def test_token_budget_raises_when_preserved_alone_exceeds_budget():
    history = [msg("system", "x" * 400, "s"), msg("user", "hi", "m0")]
    with pytest.raises(ContextBudgetExceededError):
        TokenBudgetManager(max_tokens=5, token_counter=CharTokenCounter()).prepare(
            history
        )


def test_token_budget_always_keeps_messages_without_ids():
    """Current behaviour, and a trap.

    The final filter is ``m.id in kept_ids or m.id is None``, so any message
    lacking an id bypasses the budget entirely. Model adapters return
    assistant messages with no id, so in a real run those accumulate
    unbounded — exactly what the budget exists to prevent.

    Note the budget must be large enough for the newest message, or the
    manager raises before ever reaching that filter.
    """
    counter = CharTokenCounter()
    # Five messages of 25 tokens each against a 30-token budget: selection
    # keeps exactly one, then the id-less filter puts the other four back.
    history = [Message(role="user", content="x" * 100) for _ in range(5)]
    kept = TokenBudgetManager(max_tokens=30, token_counter=counter).prepare(history)

    assert len(kept) == 5
    assert sum(counter.count_message(m) for m in kept) == 125  # budget was 30


# ---------------------------------------------------------------------------
# ToolInteractionCompactor
# ---------------------------------------------------------------------------


def test_compactor_replaces_old_interactions_with_a_summary():
    history = [msg("user", "q", "m0"), *tool_exchange(1), *tool_exchange(2)]
    kept = ToolInteractionCompactor(keep_recent=1).prepare(history)

    summaries = [
        m for m in kept if m.metadata.get("kind") == "tool_interaction_compaction"
    ]
    assert len(summaries) == 1
    assert summaries[0].role == "system"  # compaction *adds* a message
    assert "branch_hours" in summaries[0].content  # the record of what ran survives
    assert summaries[0].metadata["covered_item_ids"] == ["a1", "t1"]
    assert [m.id for m in kept if m.id] == ["m0", "a2", "t2"]


def test_compactor_covered_ids_are_empty_without_message_ids():
    """covered_item_ids is only useful when messages carry ids."""
    history = [
        Message(role="assistant", content="", metadata={"tool_calls": [{"name": "f"}]}),
        Message(role="tool", content="r"),
        *tool_exchange(2),
    ]
    kept = ToolInteractionCompactor(keep_recent=1).prepare(history)
    summary = next(
        m for m in kept if m.metadata.get("kind") == "tool_interaction_compaction"
    )

    assert summary.metadata["covered_item_ids"] == []


# ---------------------------------------------------------------------------
# SummarizingManager
# ---------------------------------------------------------------------------


def test_summarizing_below_trigger_is_a_passthrough():
    model = CountingModel()
    manager = SummarizingManager(
        model=model, keep_recent_messages=2, trigger_message_count=6
    )
    history = [msg("user", f"turn {i}", f"m{i}") for i in range(4)]

    assert manager.prepare(history) == history
    assert model.calls == 0


def test_summarizing_replaces_old_messages_with_one_summary():
    model = CountingModel()
    manager = SummarizingManager(
        model=model, keep_recent_messages=2, trigger_message_count=4
    )
    history = [msg("user", f"turn {i}", f"m{i}") for i in range(6)]

    kept = manager.prepare(history)

    assert model.calls == 1
    assert kept[0].metadata["kind"] == "conversation_summary"
    assert kept[0].role == "system"
    assert [m.id for m in kept[1:]] == ["m4", "m5"]  # keep_recent_messages
    assert kept[0].metadata["covered_item_ids"] == ["m0", "m1", "m2", "m3"]


def test_summarizing_recomputes_on_every_call():
    """prepare() is a pure view: nothing is written back, so the same history
    is summarized again on every model call rather than once per conversation.
    """
    model = CountingModel()
    manager = SummarizingManager(
        model=model, keep_recent_messages=2, trigger_message_count=4
    )
    history = [msg("user", f"turn {i}", f"m{i}") for i in range(6)]

    manager.prepare(history)
    manager.prepare(history)
    manager.prepare(history)

    assert model.calls == 3  # three identical summaries, three API calls


def test_summarizing_merges_into_a_previous_summary():
    """The merge path only runs when a prior summary is fed back in as history —
    which AgentLoop never does, since it only ever appends to ``history``.
    """
    model = CountingModel()
    manager = SummarizingManager(
        model=model, keep_recent_messages=2, trigger_message_count=4
    )
    history = [msg("user", f"turn {i}", f"m{i}") for i in range(6)]

    once = manager.prepare(history)
    twice = manager.prepare(
        [*once, msg("user", "new", "m6"), msg("user", "newer", "m7")]
    )

    summaries = [m for m in twice if m.metadata.get("kind") == "conversation_summary"]
    assert len(summaries) == 1  # merged, not accumulated


def test_summarizing_requires_a_trigger():
    with pytest.raises(ValueError):
        SummarizingManager(model=CountingModel(), keep_recent_messages=2)


def test_summarizing_keep_recent_must_be_under_trigger():
    with pytest.raises(ValueError):
        SummarizingManager(
            model=CountingModel(), keep_recent_messages=6, trigger_message_count=4
        )

from __future__ import annotations

from typing import Protocol

from sophons.models.messages import Message


# ---------------------------------------------------------------------------
# ExtractionTrigger Protocol
# ---------------------------------------------------------------------------


class ExtractionTrigger(Protocol):
    """
    Optional gate that decides whether extraction should run.

    Pass a trigger to ``LLMMemoryExtractor`` to control when the LLM
    extraction prompt is invoked.  Without a trigger, extraction runs on
    every call.

    These are convenience helpers — the caller is always free to decide
    when to call ``MemoryManager.add()`` instead.
    """

    def should_extract(self, messages: list[Message]) -> bool:
        """Return ``True`` if extraction should proceed."""
        ...


# ---------------------------------------------------------------------------
# Built-in implementations
# ---------------------------------------------------------------------------


class AlwaysTrigger:
    """Extracts on every call. Default when no trigger is configured."""

    def should_extract(self, messages: list[Message]) -> bool:
        return True


class MinMessagesTrigger:
    """
    Extracts only when the conversation has at least ``min_messages`` entries.

    Useful to avoid calling the LLM on very short conversations where
    there is nothing durable to remember yet.
    """

    def __init__(self, min_messages: int = 4) -> None:
        if min_messages < 1:
            raise ValueError("min_messages must be at least 1")
        self._min = min_messages

    def should_extract(self, messages: list[Message]) -> bool:
        return len(messages) >= self._min


class IntervalTrigger:
    """
    Extracts every ``every`` calls.

    Useful for rate-limiting expensive LLM extraction calls — e.g. extract
    every 5 turns instead of every turn.
    """

    def __init__(self, every: int = 5) -> None:
        if every < 1:
            raise ValueError("every must be at least 1")
        self._every = every
        self._count = 0

    def should_extract(self, messages: list[Message]) -> bool:
        self._count += 1
        return self._count % self._every == 0

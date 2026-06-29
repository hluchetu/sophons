from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from sophons.memory.long_term.entry import MemoryEntry
from sophons.memory.long_term.store import MemoryStore
from sophons.models.chat import AsyncChatModel
from sophons.models.messages import Message


# ---------------------------------------------------------------------------
# Reflection prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a memory synthesis assistant. You receive a set of stored memories \
and your job is to synthesize higher-level insights — patterns, generalizations, \
or abstract knowledge — that are not explicitly stated in any single memory but \
emerge from looking at them together.

Only produce insights that are genuinely durable and useful. Do not restate \
individual memories.

Respond with a JSON object in this exact format:
{
  "insights": [
    {
      "key": "<short unique slug>",
      "content": "<the insight as a clear, self-contained sentence>",
      "importance": <0.0–1.0, optional>,
      "supporting_keys": ["<key1>", "<key2>"]
    }
  ]
}

If there are no meaningful insights to synthesize, return: {"insights": []}
"""

_USER_TEMPLATE = """\
Namespace: {namespace}

Stored memories:
{memories}
"""


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class ReflectionResult:
    """
    Result of a reflection pass.

    ``entries`` are the synthesized insights that were stored.
    ``source_entry_ids`` are the IDs of the memories that were used as input.
    """

    entries: list[MemoryEntry]
    source_entry_ids: list[str]
    skipped_reason: str | None = None


# ---------------------------------------------------------------------------
# MemoryReflector
# ---------------------------------------------------------------------------


@dataclass
class MemoryReflector:
    """
    Synthesizes higher-level insights from stored memories.

    Reflection is a separate step from extraction — modeled after LangGraph's
    ``update_instructions`` pattern.  It reads existing memories, invokes an
    LLM to find patterns and generalizations, and stores the resulting
    ``MemoryEntry`` objects back with ``memory_type="semantic"``.

    The reflector tracks how many entries have been observed since the last
    reflection and how much accumulated importance they carry.  Reflection
    fires when either ``reflection_interval`` entries have been seen or the
    accumulated importance exceeds ``importance_threshold``.

    Usage::

        reflector = MemoryReflector(store=store, model=model)
        result = await reflector.observe(new_entries, namespace=("user", "alice"))
        # returns None when threshold not yet reached, ReflectionResult otherwise

        # Or trigger a reflection unconditionally:
        result = await reflector.reflect(namespace=("user", "alice"))

    Args:
        store:                 The ``MemoryStore`` to read from and write to.
        model:                 Async chat model used for synthesis.
        reflection_interval:   Number of new entries that trigger reflection.
        importance_threshold:  Accumulated importance that triggers reflection.
        source_limit:          Maximum number of existing memories to include
                               in the reflection prompt.
    """

    store: MemoryStore
    model: AsyncChatModel
    reflection_interval: int = 10
    importance_threshold: float = 3.0
    source_limit: int = 20
    _pending_count: int = field(default=0, init=False, repr=False)
    _pending_importance: float = field(default=0.0, init=False, repr=False)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def observe(
        self,
        entries: list[MemoryEntry],
        namespace: tuple[str, ...],
    ) -> ReflectionResult | None:
        """
        Accumulate ``entries`` and reflect if the threshold is reached.

        Returns ``None`` when the threshold has not been met yet.
        """
        self._pending_count += len(entries)
        self._pending_importance += sum(e.importance or 0.0 for e in entries)

        if not self._should_reflect():
            return None

        result = await self.reflect(namespace)
        self._pending_count = 0
        self._pending_importance = 0.0
        return result

    async def reflect(self, namespace: tuple[str, ...]) -> ReflectionResult:
        """
        Unconditionally run a reflection pass for ``namespace``.

        Reads up to ``source_limit`` active non-reflection memories, invokes
        the model, stores the synthesized insights, and returns the result.
        """
        sources = _select_sources(
            self.store.list(namespace=namespace),
            limit=self.source_limit,
        )

        if not sources:
            return ReflectionResult(
                entries=[],
                source_entry_ids=[],
                skipped_reason="No source memories available for reflection.",
            )

        user_content = _USER_TEMPLATE.format(
            namespace="/".join(namespace),
            memories=_format_sources(sources),
        )
        response = await self.model.invoke(
            [
                Message(role="system", content=_SYSTEM_PROMPT),
                Message(role="user", content=user_content),
            ]
        )

        try:
            payload = _parse_json(response.content)
        except ValueError:
            return ReflectionResult(
                entries=[],
                source_entry_ids=[e.id for e in sources],
                skipped_reason="Could not parse reflection response.",
            )

        entries = _build_entries(payload, namespace, sources)

        for entry in entries:
            self.store.put(entry)

        if not entries:
            return ReflectionResult(
                entries=[],
                source_entry_ids=[e.id for e in sources],
                skipped_reason="Model returned no durable insights.",
            )

        return ReflectionResult(
            entries=entries,
            source_entry_ids=[e.id for e in sources],
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _should_reflect(self) -> bool:
        return (
            self._pending_count >= self.reflection_interval
            or self._pending_importance >= self.importance_threshold
        )


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _select_sources(
    entries: list[MemoryEntry],
    limit: int,
) -> list[MemoryEntry]:
    """Top entries by importance, excluding entries already from reflection."""
    candidates = [
        e for e in entries
        if e.invalidated_at is None and e.metadata.get("source") != "reflection"
    ]
    candidates.sort(key=lambda e: (e.importance or 0.0, e.created_at), reverse=True)
    return candidates[:limit]


def _format_sources(entries: list[MemoryEntry]) -> str:
    lines = [
        f"- key={e.key} type={e.memory_type} "
        f"importance={e.importance if e.importance is not None else 'unknown'}: {e.content}"
        for e in entries
    ]
    return "\n".join(lines)


def _parse_json(text: str) -> dict[str, Any]:
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    raise ValueError("No valid JSON object found in model response.")


def _build_entries(
    payload: dict[str, Any],
    namespace: tuple[str, ...],
    sources: list[MemoryEntry],
) -> list[MemoryEntry]:
    raw_insights = payload.get("insights") or []
    if not isinstance(raw_insights, list):
        return []

    sources_by_key = {e.key: e for e in sources}
    entries: list[MemoryEntry] = []

    for raw in raw_insights:
        if not isinstance(raw, dict):
            continue

        key = _optional_str(raw.get("key"))
        content = _optional_str(raw.get("content"))
        if not key or not content:
            continue

        supporting_keys: list[str] = [
            str(k).strip()
            for k in (raw.get("supporting_keys") or [])
            if str(k).strip()
        ]
        supporting_ids = tuple(
            sources_by_key[k].id for k in supporting_keys if k in sources_by_key
        )
        importance = _optional_float(raw.get("importance"))

        entries.append(
            MemoryEntry(
                memory_type="semantic",
                namespace=namespace,
                key=key,
                content=content,
                importance=importance,
                related_ids=supporting_ids,
                metadata={
                    "source": "reflection",
                    "supporting_keys": supporting_keys,
                    "supporting_ids": list(supporting_ids),
                },
            )
        )

    return entries


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return None

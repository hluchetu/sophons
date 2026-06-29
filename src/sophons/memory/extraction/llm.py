from __future__ import annotations

import json
import re
from typing import Any

from sophons.memory.extraction.interface import (
    MemoryExtractionRequest,
    MemoryExtractionResult,
)
from sophons.memory.extraction.triggers import AlwaysTrigger, ExtractionTrigger
from sophons.memory.long_term.entry import MemoryEntry, MemoryType
from sophons.models.chat import AsyncChatModel
from sophons.models.messages import Message


# ---------------------------------------------------------------------------
# Extraction prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a memory extraction assistant. Your job is to identify durable, \
reusable facts from a conversation that are worth remembering for future \
interactions.

Extract only information that is genuinely useful across sessions — user \
preferences, goals, important entities, key decisions, or reusable knowledge. \
Skip ephemeral details, pleasantries, or single-turn facts.

Respond with a JSON object in this exact format:
{
  "records": [
    {
      "action": "create",
      "memory_type": "<semantic|entity|episodic|procedural|preference|decision>",
      "key": "<short unique slug, e.g. user.preferred_language>",
      "content": "<the memory as a clear, self-contained sentence>",
      "importance": <0.0–1.0, optional>,
      "metadata": {}
    }
  ]
}

To invalidate an outdated memory use:
  {"action": "invalidate", "key": "<existing key>"}

If nothing is worth remembering, return: {"records": []}
"""

_USER_TEMPLATE = """\
Namespace: {namespace}

Existing memories (do not duplicate these):
{existing}

Conversation:
{conversation}
"""


# ---------------------------------------------------------------------------
# LLMMemoryExtractor
# ---------------------------------------------------------------------------

_VALID_TYPES: set[str] = {
    "semantic", "entity", "episodic", "procedural", "preference", "decision"
}


class LLMMemoryExtractor:
    """
    Extracts durable ``MemoryEntry`` objects from a conversation using an LLM.

    The extractor sends the conversation history (formatted as text) and
    any existing memories in the namespace to the model, then parses the
    structured JSON response into ``MemoryEntry`` objects.

    Usage::

        extractor = LLMMemoryExtractor(model=my_model)
        result = await extractor.extract(
            MemoryExtractionRequest(
                namespace=("user", "alice"),
                messages=conversation_messages,
            )
        )
        for entry in result.entries:
            memory_store.put(entry)

    Args:
        model:   Async chat model used for extraction.
        trigger: Optional gate — if the trigger returns ``False`` the
                 extractor skips the LLM call and returns an empty result.
                 Defaults to ``AlwaysTrigger`` (always extract).
    """

    def __init__(
        self,
        model: AsyncChatModel,
        trigger: ExtractionTrigger | None = None,
    ) -> None:
        self._model = model
        self._trigger = trigger or AlwaysTrigger()

    async def extract(
        self, request: MemoryExtractionRequest
    ) -> MemoryExtractionResult:
        if not self._trigger.should_extract(request.messages):
            return MemoryExtractionResult(
                entries=[],
                skipped_reason="Extraction skipped by trigger.",
            )

        if not request.messages:
            return MemoryExtractionResult(
                entries=[],
                skipped_reason="No messages to extract from.",
            )

        user_content = _USER_TEMPLATE.format(
            namespace="/".join(request.namespace),
            existing=_format_existing(request.existing_memories),
            conversation=_format_messages(request.messages),
        )

        response = await self._model.invoke(
            [
                Message(role="system", content=_SYSTEM_PROMPT),
                Message(role="user", content=user_content),
            ]
        )

        try:
            payload = _parse_json(response.content)
        except ValueError as exc:
            return MemoryExtractionResult(
                entries=[],
                skipped_reason=f"Could not parse extraction response: {exc}",
            )

        entries, invalidated_keys = _build_entries(payload, request.namespace)
        return MemoryExtractionResult(
            entries=entries,
            invalidated_keys=invalidated_keys,
        )


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _format_messages(messages: list[Message]) -> str:
    lines: list[str] = []
    for msg in messages:
        lines.append(f"{msg.role}: {msg.content}")
    return "\n".join(lines)


def _format_existing(entries: list[MemoryEntry]) -> str:
    if not entries:
        return "None."
    lines = [
        f"- key={e.key} type={e.memory_type}: {e.content}"
        for e in entries
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _parse_json(text: str) -> dict[str, Any]:
    """Extract and parse the first JSON object from ``text``."""
    # Try the whole string first
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass
    # Extract JSON block from markdown fences or embedded text
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
) -> tuple[list[MemoryEntry], list[str]]:
    raw_records = payload.get("records") or []
    if not isinstance(raw_records, list):
        return [], []

    entries: list[MemoryEntry] = []
    invalidated_keys: list[str] = []

    for raw in raw_records:
        if not isinstance(raw, dict):
            continue

        action = str(raw.get("action", "create")).strip().lower()

        if action == "invalidate":
            key = _optional_str(raw.get("key"))
            if key:
                invalidated_keys.append(key)
            continue

        try:
            entry = _build_entry(raw, namespace)
            entries.append(entry)
        except (ValueError, KeyError):
            continue

    return entries, invalidated_keys


def _build_entry(raw: dict[str, Any], namespace: tuple[str, ...]) -> MemoryEntry:
    memory_type_raw = str(raw.get("memory_type", "semantic")).strip().lower()
    if memory_type_raw not in _VALID_TYPES:
        memory_type_raw = "semantic"
    memory_type: MemoryType = memory_type_raw  # type: ignore[assignment]

    key = _required_str(raw, "key")
    content = _required_str(raw, "content")
    importance = _optional_float(raw.get("importance"))
    metadata: dict[str, Any] = raw.get("metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {}
    metadata["extracted_by"] = "llm"

    return MemoryEntry(
        memory_type=memory_type,
        namespace=namespace,
        key=key,
        content=content,
        importance=importance,
        metadata=metadata,
    )


def _required_str(d: dict[str, Any], field: str) -> str:
    value = d.get(field)
    if value is None:
        raise ValueError(f"Missing required field: {field}")
    text = str(value).strip()
    if not text:
        raise ValueError(f"Empty required field: {field}")
    return text


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

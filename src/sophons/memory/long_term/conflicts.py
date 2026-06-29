from __future__ import annotations

from collections.abc import Iterable

from sophons.memory.long_term.entry import MemoryEntry


def _normalize(value: str) -> str:
    """Lowercase and collapse whitespace for comparison."""
    return " ".join(value.casefold().split())


def find_conflicting_entries(
    new_entry: MemoryEntry,
    candidates: Iterable[MemoryEntry],
) -> list[MemoryEntry]:
    """
    Return all active entries from ``candidates`` that conflict with
    ``new_entry``.

    Two entries conflict when they are in the same namespace, have the same
    memory type, and carry contradictory information — i.e. same key but
    different content, or same semantic subject but a different value.
    """
    conflicts: list[MemoryEntry] = []

    for candidate in candidates:
        if candidate.id == new_entry.id:
            continue
        if candidate.invalidated_at is not None:
            continue
        if candidate.namespace != new_entry.namespace:
            continue
        if candidate.memory_type != new_entry.memory_type:
            continue
        if _entries_conflict(candidate, new_entry):
            conflicts.append(candidate)

    return conflicts


def _entries_conflict(existing: MemoryEntry, new: MemoryEntry) -> bool:
    """
    Return True if ``existing`` and ``new`` carry contradictory information.

    Same key + different content always conflicts.
    Same ``subject`` metadata field + different content conflicts for
    preference and entity entries (a common pattern set by extractors).
    """
    if existing.key == new.key:
        return _normalize(existing.content) != _normalize(new.content)

    # For preference and entity types, check a shared subject/name in metadata
    if new.memory_type in ("preference", "entity"):
        existing_subject = existing.metadata.get("subject") or existing.metadata.get("name")
        new_subject = new.metadata.get("subject") or new.metadata.get("name")
        if (
            existing_subject is not None
            and new_subject is not None
            and _normalize(str(existing_subject)) == _normalize(str(new_subject))
            and _normalize(existing.content) != _normalize(new.content)
        ):
            return True

    return False

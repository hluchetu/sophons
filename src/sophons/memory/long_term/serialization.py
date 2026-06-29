from __future__ import annotations

from typing import Any

from sophons.memory.long_term.entry import MemoryEntry


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------
# MemoryEntry already has to_dict() / from_dict() on the class itself.
# These module-level functions exist so storage backends and exporters can
# call a consistent top-level API without importing the class directly.


def entry_to_dict(entry: MemoryEntry) -> dict[str, Any]:
    """Serialize a ``MemoryEntry`` to a JSON-compatible dict."""
    return entry.to_dict()


def entry_from_dict(data: dict[str, Any]) -> MemoryEntry:
    """Deserialize a ``MemoryEntry`` from a dict produced by ``entry_to_dict``."""
    return MemoryEntry.from_dict(data)


def entries_to_list(entries: list[MemoryEntry]) -> list[dict[str, Any]]:
    """Serialize a list of entries to a JSON-compatible list of dicts."""
    return [entry_to_dict(e) for e in entries]


def entries_from_list(data: list[dict[str, Any]]) -> list[MemoryEntry]:
    """Deserialize a list of dicts into a list of ``MemoryEntry`` objects."""
    return [entry_from_dict(d) for d in data]

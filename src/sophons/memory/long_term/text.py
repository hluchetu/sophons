from __future__ import annotations

from sophons.memory.long_term.entry import MemoryEntry


def searchable_text(entry: MemoryEntry) -> str:
    """
    Build a single searchable string from a ``MemoryEntry``.

    Combines the key, content, memory type, namespace, and any string
    metadata values into one string suitable for lexical search or
    embedding.

    The namespace and key provide context so retrievers can distinguish
    entries that share similar content but belong to different owners.
    """
    namespace_str = " ".join(entry.namespace)
    metadata_values = " ".join(
        str(v) for v in entry.metadata.values() if isinstance(v, (str, int, float))
    )
    parts = [
        namespace_str,
        entry.memory_type,
        entry.key,
        entry.content,
        metadata_values,
    ]
    return " ".join(p for p in parts if p.strip())

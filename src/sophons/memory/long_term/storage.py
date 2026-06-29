from __future__ import annotations

from typing import Protocol

from sophons.memory.long_term.entry import MemoryEntry, MemoryType


# ---------------------------------------------------------------------------
# MemoryStorage Protocol
# ---------------------------------------------------------------------------


class MemoryStorage(Protocol):
    """
    Contract for long-term memory storage backends.

    Sophons defines this contract. You bring the backend — SQLite, Postgres,
    Redis, a cloud API, or anything else. The only built-in implementation is
    ``InMemoryStorage``, which is suitable for tests and development.

    To use a custom backend implement this Protocol in your own class and
    pass it to ``MemoryStore``::

        class MyStorage:
            def put(self, entry: MemoryEntry) -> None: ...
            def get(self, namespace, key): ...
            ...

        store = MemoryStore(storage=MyStorage())

    All methods are synchronous. For async storage backends wrap your calls
    with ``asyncio.to_thread`` or provide an async adapter layer.
    """

    def put(self, entry: MemoryEntry) -> None:
        """
        Store an entry.

        If an entry with the same ``namespace`` + ``key`` already exists it is
        replaced.
        """
        ...

    def get(
        self,
        namespace: tuple[str, ...],
        key: str,
    ) -> MemoryEntry | None:
        """Return the entry for ``namespace`` + ``key``, or ``None``."""
        ...

    def get_by_id(self, entry_id: str) -> MemoryEntry | None:
        """Return the entry with the given ``id``, or ``None``."""
        ...

    def get_many(self, ids: list[str]) -> list[MemoryEntry]:
        """Return all entries whose ``id`` is in ``ids``."""
        ...

    def list(
        self,
        namespace: tuple[str, ...],
        memory_type: MemoryType | None = None,
        include_invalidated: bool = False,
    ) -> list[MemoryEntry]:
        """
        Return all entries in ``namespace``.

        Args:
            namespace:           The namespace to list.
            memory_type:         If given, return only entries of this type.
            include_invalidated: If ``False`` (default) invalidated entries
                                 are excluded.
        """
        ...

    def delete(
        self,
        namespace: tuple[str, ...],
        key: str,
    ) -> None:
        """Remove the entry for ``namespace`` + ``key``."""
        ...


# ---------------------------------------------------------------------------
# InMemoryStorage
# ---------------------------------------------------------------------------


class InMemoryStorage:
    """
    Dict-backed storage. No dependencies, no setup.

    Suitable for tests, prototypes, and single-process applications where
    persistence across restarts is not required. All data is lost when the
    process exits.

    Satisfies the ``MemoryStorage`` Protocol.
    """

    def __init__(self) -> None:
        # Primary index: (namespace, key) → entry
        self._by_key: dict[tuple[tuple[str, ...], str], MemoryEntry] = {}
        # Secondary index: id → entry
        self._by_id: dict[str, MemoryEntry] = {}

    def put(self, entry: MemoryEntry) -> None:
        lookup = (entry.namespace, entry.key)
        previous = self._by_key.get(lookup)
        if previous is not None:
            self._by_id.pop(previous.id, None)
        self._by_key[lookup] = entry
        self._by_id[entry.id] = entry

    def get(
        self,
        namespace: tuple[str, ...],
        key: str,
    ) -> MemoryEntry | None:
        return self._by_key.get((namespace, key))

    def get_by_id(self, entry_id: str) -> MemoryEntry | None:
        return self._by_id.get(entry_id)

    def get_many(self, ids: list[str]) -> list[MemoryEntry]:
        return [self._by_id[i] for i in ids if i in self._by_id]

    def list(
        self,
        namespace: tuple[str, ...],
        memory_type: MemoryType | None = None,
        include_invalidated: bool = False,
    ) -> list[MemoryEntry]:
        entries = [
            entry
            for (ns, _), entry in self._by_key.items()
            if ns == namespace
        ]
        if memory_type is not None:
            entries = [e for e in entries if e.memory_type == memory_type]
        if not include_invalidated:
            entries = [e for e in entries if e.invalidated_at is None]
        return entries

    def delete(
        self,
        namespace: tuple[str, ...],
        key: str,
    ) -> None:
        entry = self._by_key.pop((namespace, key), None)
        if entry is not None:
            self._by_id.pop(entry.id, None)

    def count(self, namespace: tuple[str, ...] | None = None) -> int:
        """Return the number of active entries, optionally scoped to a namespace."""
        if namespace is None:
            return sum(1 for e in self._by_id.values() if e.invalidated_at is None)
        return len(self.list(namespace))

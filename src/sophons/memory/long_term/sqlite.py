from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from sophons.memory.long_term.entry import MemoryEntry, MemoryType


class SQLiteMemoryStorage:
    """
    SQLite-backed long-term memory storage.

    Entries are stored as JSON so the backend stays aligned with
    ``MemoryEntry.to_dict()`` while SQLite indexes the fields needed for common
    lookups. The implementation is synchronous to match ``MemoryStorage``.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._path)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def put(self, entry: MemoryEntry) -> None:
        namespace = _namespace_key(entry.namespace)
        payload = json.dumps(entry.to_dict(), ensure_ascii=False)
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO memory_entries (
                    id, namespace, key, memory_type, invalidated_at, data
                )
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    namespace = excluded.namespace,
                    key = excluded.key,
                    memory_type = excluded.memory_type,
                    invalidated_at = excluded.invalidated_at,
                    data = excluded.data
                """,
                (
                    entry.id,
                    namespace,
                    entry.key,
                    entry.memory_type,
                    entry.invalidated_at.isoformat()
                    if entry.invalidated_at is not None
                    else None,
                    payload,
                ),
            )

    def get(
        self,
        namespace: tuple[str, ...],
        key: str,
    ) -> MemoryEntry | None:
        row = self._conn.execute(
            """
            SELECT data
            FROM memory_entries
            WHERE namespace = ? AND key = ? AND invalidated_at IS NULL
            ORDER BY created_at DESC, rowid DESC
            LIMIT 1
            """,
            (_namespace_key(namespace), key),
        ).fetchone()
        return _entry_from_row(row)

    def get_by_id(self, entry_id: str) -> MemoryEntry | None:
        row = self._conn.execute(
            "SELECT data FROM memory_entries WHERE id = ?",
            (entry_id,),
        ).fetchone()
        return _entry_from_row(row)

    def get_many(self, ids: list[str]) -> list[MemoryEntry]:
        entries: list[MemoryEntry] = []
        for entry_id in ids:
            entry = self.get_by_id(entry_id)
            if entry is not None:
                entries.append(entry)
        return entries

    def list(
        self,
        namespace: tuple[str, ...],
        memory_type: MemoryType | None = None,
        include_invalidated: bool = False,
    ) -> list[MemoryEntry]:
        clauses = ["namespace = ?"]
        params: list[object] = [_namespace_key(namespace)]
        if memory_type is not None:
            clauses.append("memory_type = ?")
            params.append(memory_type)
        if not include_invalidated:
            clauses.append("invalidated_at IS NULL")

        rows = self._conn.execute(
            f"""
            SELECT data
            FROM memory_entries
            WHERE {' AND '.join(clauses)}
            ORDER BY created_at ASC, rowid ASC
            """,
            params,
        ).fetchall()
        entries = [_entry_from_row(row) for row in rows]
        return [entry for entry in entries if entry is not None]

    def delete(
        self,
        namespace: tuple[str, ...],
        key: str,
    ) -> None:
        with self._conn:
            self._conn.execute(
                "DELETE FROM memory_entries WHERE namespace = ? AND key = ?",
                (_namespace_key(namespace), key),
            )

    def close(self) -> None:
        self._conn.close()

    def _init_schema(self) -> None:
        with self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_entries (
                    id TEXT PRIMARY KEY,
                    namespace TEXT NOT NULL,
                    key TEXT NOT NULL,
                    memory_type TEXT NOT NULL,
                    invalidated_at TEXT,
                    data TEXT NOT NULL,
                    created_at TEXT GENERATED ALWAYS AS
                        (json_extract(data, '$.created_at')) VIRTUAL
                )
                """
            )
            self._conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_memory_entries_lookup
                ON memory_entries(namespace, key, invalidated_at)
                """
            )
            self._conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_memory_entries_type
                ON memory_entries(namespace, memory_type, invalidated_at)
                """
            )


def _namespace_key(namespace: tuple[str, ...]) -> str:
    return json.dumps(list(namespace), separators=(",", ":"))


def _entry_from_row(row: sqlite3.Row | None) -> MemoryEntry | None:
    if row is None:
        return None
    return MemoryEntry.from_dict(json.loads(row["data"]))

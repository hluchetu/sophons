from __future__ import annotations

from sophons.memory import MemoryEntry, SQLiteMemoryStorage


def test_sqlite_memory_storage_persists_entries(tmp_path) -> None:
    path = tmp_path / "memory.sqlite"
    storage = SQLiteMemoryStorage(path)
    entry = MemoryEntry(
        memory_type="preference",
        namespace=("user", "alice"),
        key="style",
        content="User prefers detailed notes.",
    )

    storage.put(entry)
    storage.close()

    reopened = SQLiteMemoryStorage(path)

    assert reopened.get(("user", "alice"), "style") == entry
    assert reopened.get_by_id(entry.id) == entry
    assert reopened.list(("user", "alice")) == [entry]
    reopened.close()

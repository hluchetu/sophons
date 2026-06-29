from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Protocol

from sophons.models.messages import Message

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SessionManager Protocol
# ---------------------------------------------------------------------------


class SessionManager(Protocol):
    """
    Persists and restores conversation history across agent runs.

    The agent loop calls ``load`` before a run to seed the message history,
    and ``save`` after a run to persist the updated history.

    Both methods are async so implementations can use any storage backend
    — in-memory, file system, database, or a remote API — without blocking
    the event loop.
    """

    async def load(self, session_id: str) -> list[Message]:
        """
        Return the message history for ``session_id``.

        Returns an empty list if no session exists yet.
        """
        ...

    async def save(self, session_id: str, messages: list[Message]) -> None:
        """
        Persist ``messages`` under ``session_id``, replacing any prior state.
        """
        ...

    async def delete(self, session_id: str) -> None:
        """
        Delete the session for ``session_id``.

        Does nothing if the session does not exist.
        """
        ...

    async def exists(self, session_id: str) -> bool:
        """Return True if a session exists for ``session_id``."""
        ...


# ---------------------------------------------------------------------------
# InMemorySessionManager
# ---------------------------------------------------------------------------


class InMemorySessionManager:
    """
    Stores sessions in a plain dict.

    Fast and requires no setup. Sessions are lost when the process exits.
    Useful for testing and short-lived single-process applications.

    All methods are async to satisfy the SessionManager Protocol even though
    no I/O is performed.
    """

    def __init__(self) -> None:
        self._store: dict[str, list[Message]] = {}

    async def load(self, session_id: str) -> list[Message]:
        messages = self._store.get(session_id, [])
        logger.debug("session=%s loaded %d messages", session_id, len(messages))
        return list(messages)

    async def save(self, session_id: str, messages: list[Message]) -> None:
        self._store[session_id] = list(messages)
        logger.debug("session=%s saved %d messages", session_id, len(messages))

    async def delete(self, session_id: str) -> None:
        self._store.pop(session_id, None)
        logger.debug("session=%s deleted", session_id)

    async def exists(self, session_id: str) -> bool:
        return session_id in self._store

    def session_ids(self) -> list[str]:
        """Return all active session IDs. Convenience method for inspection."""
        return list(self._store.keys())


# ---------------------------------------------------------------------------
# FileSessionManager
# ---------------------------------------------------------------------------


class FileSessionManager:
    """
    Persists sessions as JSON files on disk.

    Each session is stored as a single JSON file at::

        {directory}/{session_id}.json

    Sessions survive process restarts. Multiple processes sharing the same
    directory should not write the same session concurrently — no locking
    is provided.

    Args:
        directory: Path to the directory where session files are stored.
                   Created automatically if it does not exist.
    """

    def __init__(self, directory: str | Path) -> None:
        self._dir = Path(directory)
        self._dir.mkdir(parents=True, exist_ok=True)

    async def load(self, session_id: str) -> list[Message]:
        path = self._path(session_id)
        if not path.exists():
            return []
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            messages = [Message.from_dict(m) for m in data]
            logger.debug(
                "session=%s path=%s loaded %d messages",
                session_id,
                path,
                len(messages),
            )
            return messages
        except Exception as exc:
            logger.warning(
                "session=%s path=%s failed to load: %r — returning empty history",
                session_id,
                path,
                exc,
            )
            return []

    async def save(self, session_id: str, messages: list[Message]) -> None:
        path = self._path(session_id)
        data = [m.to_dict() for m in messages]
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.debug(
            "session=%s path=%s saved %d messages",
            session_id,
            path,
            len(messages),
        )

    async def delete(self, session_id: str) -> None:
        path = self._path(session_id)
        if path.exists():
            path.unlink()
            logger.debug("session=%s path=%s deleted", session_id, path)

    async def exists(self, session_id: str) -> bool:
        return self._path(session_id).exists()

    def _path(self, session_id: str) -> Path:
        # Sanitise the session_id so it is safe as a filename
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in session_id)
        return self._dir / f"{safe}.json"

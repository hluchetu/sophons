from __future__ import annotations

import json
import logging
import os
import stat
import tempfile
from pathlib import Path
from typing import Protocol

from sophons.agents.control import call_callable
from sophons.errors import SessionPersistenceError
from sophons.models.messages import Message

logger = logging.getLogger(__name__)


class SessionManager(Protocol):
    """Async session persistence. Missing history is empty; storage errors must raise."""

    async def load(self, session_id: str) -> list[Message]: ...
    async def save(self, session_id: str, messages: list[Message]) -> None: ...
    async def delete(self, session_id: str) -> None: ...
    async def exists(self, session_id: str) -> bool: ...


class InMemorySessionManager:
    """Process-local history, lost on restart; not durable application storage."""

    def __init__(self) -> None:
        self._store: dict[str, list[Message]] = {}

    async def load(self, session_id: str) -> list[Message]:
        return list(self._store.get(session_id, []))

    async def save(self, session_id: str, messages: list[Message]) -> None:
        self._store[session_id] = list(messages)

    async def delete(self, session_id: str) -> None:
        self._store.pop(session_id, None)

    async def exists(self, session_id: str) -> bool:
        return session_id in self._store

    def session_ids(self) -> list[str]:
        return list(self._store.keys())


class FileSessionManager:
    """JSON history with atomic file replacement and explicit storage failures.

    No cross-process revision locking is provided: concurrent writers are still
    last-writer-wins. Applications needing concurrency control should use a
    transactional repository. File fsync does not promise complete power-loss
    durability on every filesystem. No plaintext session contents are logged.
    """

    def __init__(self, directory: str | Path) -> None:
        self._dir = Path(directory)
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise SessionPersistenceError("initialize") from error

    async def load(self, session_id: str) -> list[Message]:
        return await call_callable(self._load_file, session_id)

    def _load_file(self, session_id: str) -> list[Message]:
        try:
            self._check_directory("load")
            text = self._path(session_id).read_text(encoding="utf-8")
        except FileNotFoundError:
            self._check_directory("load")
            return []  # A genuinely new session, not an unavailable store.
        except Exception as error:
            raise SessionPersistenceError("load") from error
        try:
            data = json.loads(text)
            if not isinstance(data, list):
                raise ValueError("Session history must be a list")
            messages = [Message.from_dict(item) for item in data]
            if any(message.role not in {"system", "user", "assistant", "tool"}
                   or not isinstance(message.content, str)
                   or not isinstance(message.metadata, dict)
                   for message in messages):
                raise ValueError("Invalid stored message")
            return messages
        except Exception as error:
            raise SessionPersistenceError("load") from error

    async def save(self, session_id: str, messages: list[Message]) -> None:
        await call_callable(self._save_file, session_id, messages)

    def _save_file(self, session_id: str, messages: list[Message]) -> None:
        temporary: Path | None = None
        try:
            body = json.dumps([message.to_dict() for message in messages], indent=2, ensure_ascii=False)
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self._dir,
                    prefix=".sophons-session-", suffix=".tmp", delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(body)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self._path(session_id))
        except Exception as error:
            raise SessionPersistenceError("save") from error
        finally:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    logger.warning("Session temporary-file cleanup failed")

    async def delete(self, session_id: str) -> None:
        await call_callable(self._delete_file, session_id)

    def _delete_file(self, session_id: str) -> None:
        try:
            self._check_directory("delete")
            self._path(session_id).unlink()
        except FileNotFoundError:
            self._check_directory("delete")
        except Exception as error:
            raise SessionPersistenceError("delete") from error

    async def exists(self, session_id: str) -> bool:
        return await call_callable(self._exists_file, session_id)

    def _exists_file(self, session_id: str) -> bool:
        try:
            self._check_directory("exists")
            self._path(session_id).stat()
            return True
        except FileNotFoundError:
            self._check_directory("exists")
            return False
        except Exception as error:
            raise SessionPersistenceError("exists") from error

    def _check_directory(self, operation: str) -> None:
        try:
            if not stat.S_ISDIR(self._dir.stat().st_mode):
                raise OSError("Session store is not a directory")
        except OSError as error:
            raise SessionPersistenceError(operation) from error

    def _path(self, session_id: str) -> Path:
        safe = "".join(char if char.isalnum() or char in "-_." else "_" for char in session_id)
        return self._dir / f"{safe}.json"

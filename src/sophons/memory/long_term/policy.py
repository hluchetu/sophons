from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


# ---------------------------------------------------------------------------
# NamespacePolicy Protocol
# ---------------------------------------------------------------------------


class NamespacePolicy(Protocol):
    """
    Controls read and write access to memory namespaces.

    Implement this Protocol to restrict which namespaces an agent or
    component is allowed to read from or write to.

    Example — allow a component to read all namespaces but only write to
    its own::

        class ReadOnlyPolicy:
            def can_read(self, namespace): return True
            def can_write(self, namespace): return False
    """

    def can_read(self, namespace: tuple[str, ...]) -> bool:
        """Return True if reading from ``namespace`` is permitted."""
        ...

    def can_write(self, namespace: tuple[str, ...]) -> bool:
        """Return True if writing to ``namespace`` is permitted."""
        ...


# ---------------------------------------------------------------------------
# Built-in implementations
# ---------------------------------------------------------------------------


class AllowAllPolicy:
    """Permits all reads and writes. Default when no policy is configured."""

    def can_read(self, namespace: tuple[str, ...]) -> bool:
        return True

    def can_write(self, namespace: tuple[str, ...]) -> bool:
        return True


class DenyAllPolicy:
    """Denies all reads and writes. Useful as a safe default in tests."""

    def can_read(self, namespace: tuple[str, ...]) -> bool:
        return False

    def can_write(self, namespace: tuple[str, ...]) -> bool:
        return False


@dataclass(frozen=True)
class NamespacePrefixPolicy:
    """
    Permits access only to namespaces that start with ``allowed_prefix``.

    Example — allow access only to namespaces under ``("user", "alice")``::

        policy = NamespacePrefixPolicy(allowed_prefix=("user", "alice"))
    """

    allowed_prefix: tuple[str, ...]
    allow_reads: bool = True
    allow_writes: bool = True

    def can_read(self, namespace: tuple[str, ...]) -> bool:
        return self.allow_reads and self._matches(namespace)

    def can_write(self, namespace: tuple[str, ...]) -> bool:
        return self.allow_writes and self._matches(namespace)

    def _matches(self, namespace: tuple[str, ...]) -> bool:
        return namespace[: len(self.allowed_prefix)] == self.allowed_prefix

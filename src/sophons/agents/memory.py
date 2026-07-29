from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

NamespaceResolver = Callable[[str, str | None], tuple[str, ...]]


@dataclass(frozen=True, kw_only=True)
class MemoryConfig:
    """
    Controls how an Agent uses long-term memory.

    The defaults match the safest high-level behavior: retrieve relevant
    memories before each run and extract durable memories after a successful
    model response. Tool access is opt-in so applications choose whether the
    model may search or write memory directly.
    """

    namespace: tuple[str, ...] | None = None
    namespace_resolver: NamespaceResolver | None = None
    inject: bool = True
    inject_limit: int = 5
    extract_after_run: bool = True
    add_search_tool: bool = False
    add_write_tool: bool = False
    context_header: str = "Relevant long-term memory:"

    def resolve_namespace(
        self,
        input: str,
        session_id: str | None,
    ) -> tuple[str, ...] | None:
        if self.namespace_resolver is not None:
            return self.namespace_resolver(input, session_id)
        if self.namespace is not None:
            return self.namespace
        if session_id:
            return ("session", session_id)
        return None

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

GuardrailAction = Literal["allow", "block", "transform"]

Boundary = Literal["input", "tool", "output"]


@dataclass(frozen=True, slots=True)
class GuardrailContext:
    """Where in the run a check is happening, and for whom."""

    boundary: Boundary
    session_id: str | None = None
    tool_name: str | None = None  # set at the tool boundary
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class GuardrailDecision:
    """The structured outcome of one guardrail check."""

    action: GuardrailAction = "allow"
    reason: str = ""
    transformed: Any | None = None  # replacement value when action="transform"
    message: str | None = None  # safe user-facing text when action="block"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def allowed(self) -> bool:
        return self.action != "block"

    @classmethod
    def allow(cls) -> GuardrailDecision:
        return cls()

    @classmethod
    def block(cls, reason: str, *, message: str | None = None) -> GuardrailDecision:
        return cls(action="block", reason=reason, message=message)

    @classmethod
    def transform(cls, value: Any, *, reason: str) -> GuardrailDecision:
        return cls(action="transform", transformed=value, reason=reason)


@runtime_checkable
class Guardrail(Protocol):
    """Guardrail contract: one value at one boundary in, one decision out."""

    @property
    def name(self) -> str: ...

    async def check(
        self, value: Any, *, context: GuardrailContext
    ) -> GuardrailDecision: ...

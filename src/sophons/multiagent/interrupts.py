"""Interrupt types for pausing and resuming multi-agent execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4


@dataclass(frozen=True, slots=True)
class Interrupt:
    """A request that pauses a Graph or Swarm execution.

    An interrupt is commonly used when an agent needs human input,
    approval, clarification, or an external event before execution can
    continue. Its ID matches a resume value to the original request.
    """

    value: Any
    id: str = field(default_factory=lambda: str(uuid4()))

    def to_dict(self) -> dict[str, Any]:
        """Convert the interrupt into serializable state."""

        return {
            "id": self.id,
            "value": self.value,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Interrupt:
        """Restore an interrupt from serialized state."""

        interrupt_id = data.get("id")

        if not isinstance(interrupt_id, str) or not interrupt_id.strip():
            raise ValueError("Interrupt state must contain a non-empty 'id'.")

        return cls(
            id=interrupt_id,
            value=data.get("value"),
        )

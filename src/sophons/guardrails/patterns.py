from __future__ import annotations

import re
from typing import Any, Literal

from sophons.guardrails.base import Boundary, GuardrailContext, GuardrailDecision

# Common patterns, ready to import. Deliberately conservative: better to
# miss an exotic format than to redact half the conversation.
CREDIT_CARD = r"\b\d{4}[ -]?\d{4}[ -]?\d{4}[ -]?\d{4}\b"
US_SSN = r"\b\d{3}-\d{2}-\d{4}\b"
EMAIL = r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"

PatternAction = Literal["block", "redact"]


class PatternGuardrail:
    """
    Regex checks over text crossing a boundary — the PII / leak workhorse.

    Args:
        patterns:    Named patterns, ``{"card": CREDIT_CARD, ...}``. The
                     label is what appears in reasons and traces — never
                     the matched text itself.
        action:      ``"redact"`` (default) transforms the value with all
                     matches replaced; ``"block"`` stops it entirely.
        boundaries:  Where to look. Defaults to input and output; add
                     ``"tool"`` to scan stringified tool arguments.
        replacement: The redaction text.
        message:     User-facing text when blocking.
        name:        Trace name — give each instance a specific one
                     (``"pattern:pii"``) when running several.
    """

    def __init__(
        self,
        *,
        patterns: dict[str, str | re.Pattern[str]],
        action: PatternAction = "redact",
        boundaries: tuple[Boundary, ...] = ("input", "output"),
        replacement: str = "[redacted]",
        message: str | None = None,
        name: str = "pattern",
    ) -> None:
        self.patterns = {
            label: re.compile(p) if isinstance(p, str) else p
            for label, p in patterns.items()
        }
        self.action: PatternAction = action
        self.boundaries = boundaries
        self.replacement = replacement
        self.message = message
        self.name = name

    async def check(
        self, value: Any, *, context: GuardrailContext
    ) -> GuardrailDecision:
        if context.boundary not in self.boundaries:
            return GuardrailDecision.allow()

        text = str(value)
        hits = [label for label, rx in self.patterns.items() if rx.search(text)]
        if not hits:
            return GuardrailDecision.allow()

        found = ", ".join(hits)
        if self.action == "block":
            return GuardrailDecision.block(
                f"matched {found} at {context.boundary}", message=self.message
            )

        redacted = text
        for rx in self.patterns.values():
            redacted = rx.sub(self.replacement, redacted)
        return GuardrailDecision.transform(
            redacted, reason=f"redacted {found} at {context.boundary}"
        )

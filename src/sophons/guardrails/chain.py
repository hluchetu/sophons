from __future__ import annotations

from typing import Any, Literal

from opentelemetry import trace

from sophons.guardrails.base import Guardrail, GuardrailContext, GuardrailDecision

ChainMode = Literal["enforce", "shadow"]


class GuardrailChain:
    """
    Runs guardrails in order against one value at one boundary.

    Semantics:
    - ``allow``     -> continue to the next guardrail.
    - ``transform`` -> the transformed value feeds the next guardrail
                       (and the pipeline, in enforce mode).
    - ``block``     -> stop at the first block (enforce mode) and return it.

    In ``shadow`` mode nothing is enforced: every guardrail still runs and
    every non-allow decision is recorded — on the active span and in the
    returned decision's metadata — but the chain always returns allow with
    the original value untouched. Shadow is the tuning phase: measure the
    false-positive rate before users pay for it.
    """

    def __init__(
        self,
        guardrails: list[Guardrail],
        *,
        mode: ChainMode = "enforce",
    ) -> None:
        self.guardrails = guardrails
        self.mode: ChainMode = mode

    async def check(
        self, value: Any, *, context: GuardrailContext
    ) -> GuardrailDecision:
        current = value
        records: list[dict[str, Any]] = []

        for guardrail in self.guardrails:
            decision = await guardrail.check(current, context=context)
            if decision.action == "allow":
                continue

            records.append(
                {
                    "guardrail": guardrail.name,
                    "action": decision.action,
                    "reason": decision.reason,
                }
            )
            self._record_on_span(guardrail.name, decision, context)

            if decision.action == "transform" and self.mode == "enforce":
                current = decision.transformed
            elif decision.action in ("block", "confirm") and self.mode == "enforce":
                # Both stop the chain: block is final; confirm is "not
                # allowed until an approver upgrades it" — the loop decides.
                return GuardrailDecision(
                    action=decision.action,
                    reason=decision.reason,
                    message=decision.message,
                    metadata={"guardrail": guardrail.name, "decisions": records},
                )

        if self.mode == "enforce" and current is not value:
            return GuardrailDecision(
                action="transform",
                transformed=current,
                reason="; ".join(r["reason"] for r in records),
                metadata={"decisions": records},
            )
        return GuardrailDecision(metadata={"decisions": records})

    def _record_on_span(
        self, name: str, decision: GuardrailDecision, context: GuardrailContext
    ) -> None:
        """Every non-allow decision becomes a span event — shadow included.
        The flight recorder logs the near-misses."""
        trace.get_current_span().add_event(
            "guardrail_decision",
            {
                "sophons.guardrail.name": name,
                "sophons.guardrail.action": decision.action,
                "sophons.guardrail.boundary": context.boundary,
                "sophons.guardrail.reason": decision.reason,
                "sophons.guardrail.mode": self.mode,
            },
        )

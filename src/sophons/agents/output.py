"""Structured output through the tool-calling channel.

A model will happily return prose when you wanted a record. Getting it to
fill a fixed shape reliably has three possible approaches:

1. **Tool calling** — declare a tool whose *argument schema* is the shape
   you want back. There is no function behind it; asking the model to
   "call" it is really asking it to fill in that schema, and the model's
   own tool-calling machinery does the enforcing.
2. **Native structured output** — hand the provider a JSON schema via
   ``response_format`` and let it constrain decoding.
3. **Prompted output** — describe the schema in the prompt and parse
   whatever text comes back.

Sophons uses (1), and the choice is forced rather than aesthetic. DeepSeek,
the model these examples run on, rejects
``response_format={"type": "json_schema"}`` as unavailable, but does honour
JSON Schema inside tool definitions — which rules out (2). Option (3) is
unreliable precisely because nothing obliges the model to comply; you
discover it drifted only when parsing fails.

Tool calling is also the most portable of the three, with near-universal
model support, and it reuses machinery Sophons already has for real tools —
so structured output costs one class rather than a parallel code path. The
major agent frameworks land in the same place for the same reasons; see
Pydantic AI's "Tool Output" mode and LangChain's ``function_calling``
method.

Validation failures are handed back to the model as a tool error so it can
correct itself instead of failing the run outright. Those retries are
bounded by the run's ``max_steps`` limit rather than a separate budget.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ValidationError

OUTPUT_TOOL_NAME = "structured_response"


class OutputTool:
    """
    A tool that exists only to carry the shape of the final answer.

    Unlike a real tool it is never executed: the agent loop intercepts the
    call, validates the arguments into ``output_type``, and ends the run.
    ``call`` exists only so this class satisfies the ``Tool`` protocol and
    can live in the same registry as everything else.
    """

    def __init__(
        self,
        output_type: type[BaseModel],
        name: str = OUTPUT_TOOL_NAME,
    ) -> None:
        self._output_type = output_type
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        # The output model's own docstring describes what the model is being
        # asked to produce better than anything generated here, so prefer it.
        doc = (self._output_type.__doc__ or "").strip()
        return doc or (
            f"Return the final answer as {self._output_type.__name__}. "
            "Call this once you can fill in every required field."
        )

    @property
    def args_schema(self) -> dict[str, Any]:
        # Pydantic already emits JSON Schema, so there is nothing to
        # hand-roll here — unlike @tool, which reads type hints off a plain
        # function to build the same thing.
        return self._output_type.model_json_schema()

    def validate(self, args: dict[str, Any]) -> BaseModel:
        """Turn the model's arguments into the output object, or raise."""
        return self._output_type.model_validate(args)

    def call(self, args: dict[str, Any]) -> dict[str, Any]:
        # Unreachable in normal operation: AgentLoop intercepts this tool
        # before execution. Present to satisfy the Tool protocol.
        return dict(args)


def format_validation_error(error: ValidationError) -> str:
    """
    Render a ValidationError as instructions the model can act on.

    Pydantic's default string form is aimed at developers — it carries
    documentation URLs and input snippets. The model needs the shortest
    accurate statement of what was wrong with which field.
    """
    lines = [
        "Your arguments did not match the required schema. "
        "Fix these problems and call the tool again:"
    ]
    for item in error.errors():
        location = ".".join(str(part) for part in item["loc"]) or "(root)"
        lines.append(f"- {location}: {item['msg']}")
    return "\n".join(lines)

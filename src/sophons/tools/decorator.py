from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, get_args, get_origin, get_type_hints

from sophons.tools.base import ToolArgs, ToolResult, ToolSchema


@dataclass(frozen=True, slots=True)
class FunctionTool:
    """Tool wrapper built from a normal Python function."""

    name: str
    description: str
    args_schema: ToolSchema
    fn: Callable[..., Any]

    def call(self, args: ToolArgs) -> ToolResult:
        result = self.fn(**args)
        if isinstance(result, dict):
            return result
        return {"result": result}


def tool(fn: Callable[..., Any]) -> FunctionTool:
    """Convert a Python function into a Sophons tool."""

    return FunctionTool(
        name=fn.__name__,
        description=inspect.getdoc(fn) or "",
        args_schema=build_args_schema(fn),
        fn=fn,
    )


def build_args_schema(fn: Callable[..., Any]) -> ToolSchema:
    """Build a JSON-schema-like argument schema from function type hints."""

    signature = inspect.signature(fn)
    type_hints = get_type_hints(fn)

    properties: dict[str, Any] = {}
    required: list[str] = []

    for name, parameter in signature.parameters.items():
        if parameter.kind in {
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        }:
            continue

        parameter_type = type_hints.get(name, str)
        schema = _python_type_to_json_schema(parameter_type)

        if parameter.default is inspect.Parameter.empty:
            required.append(name)
        else:
            schema["default"] = parameter.default

        properties[name] = schema

    return {
        "type": "object",
        "properties": properties,
        "required": required,
    }


def _python_type_to_json_schema(annotation: Any) -> dict[str, Any]:
    origin = get_origin(annotation)
    args = get_args(annotation)

    if annotation is str:
        return {"type": "string"}
    if annotation is int:
        return {"type": "integer"}
    if annotation is float:
        return {"type": "number"}
    if annotation is bool:
        return {"type": "boolean"}
    if annotation is dict or origin is dict:
        return {"type": "object"}
    if annotation is list or origin is list:
        item_schema = _python_type_to_json_schema(args[0]) if args else {}
        return {"type": "array", "items": item_schema}

    return {"type": "string"}

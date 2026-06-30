from __future__ import annotations

from sophons.tools import FunctionTool, Tool, build_args_schema, tool


def test_tool_decorator_wraps_function() -> None:
    @tool
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    assert isinstance(add, FunctionTool)
    assert isinstance(add, Tool)
    assert add.name == "add"
    assert add.description == "Add two numbers."
    assert add.args_schema == {
        "type": "object",
        "properties": {
            "a": {"type": "integer"},
            "b": {"type": "integer"},
        },
        "required": ["a", "b"],
    }
    assert add.call({"a": 2, "b": 3}) == {"result": 5}


def test_tool_decorator_preserves_dict_results() -> None:
    @tool
    def search_docs(query: str, limit: int = 5) -> dict:
        """Search documents."""
        return {"query": query, "limit": limit, "documents": []}

    assert search_docs.args_schema == {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "limit": {"type": "integer", "default": 5},
        },
        "required": ["query"],
    }
    assert search_docs.call({"query": "tools"}) == {
        "query": "tools",
        "limit": 5,
        "documents": [],
    }


def test_build_args_schema_supports_lists() -> None:
    def summarize(items: list[str]) -> str:
        return ", ".join(items)

    assert build_args_schema(summarize) == {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "required": ["items"],
    }

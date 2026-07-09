---
title: Tools
description: Tools are Python functions that Sophons agents can call.
---

# Tools

Tools are Python functions an agent can call.

Use tools when the model needs to do something outside its own text generation:
search documents, read memory, call an API, query a database, or run a small
calculation.

The easiest way to create a tool is to decorate a normal Python function with
`@tool`.

```python
from sophons.agents import Agent
from sophons.tools import tool


@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


agent = Agent(
    model=my_model,
    tools=[add],
)

result = agent.run_sync("What is 2 + 3?")
print(result.message)
```

## How It Works

`@tool` turns your Python function into a `FunctionTool`.

Sophons reads the function and builds a tool definition from it.

| Function part | Tool field |
|---|---|
| Function name | Tool name |
| Docstring | Tool description |
| Type hints | Argument schema |
| Default values | Optional arguments |
| Function body | Tool implementation |

This function:

```python
@tool
def search_docs(query: str, limit: int = 5) -> dict:
    """Search documents."""
    return {
        "query": query,
        "limit": limit,
        "documents": [],
    }
```

becomes a tool named `search_docs`.

Sophons infers this argument schema:

```python
{
    "type": "object",
    "properties": {
        "query": {"type": "string"},
        "limit": {"type": "integer", "default": 5},
    },
    "required": ["query"],
}
```

`query` is required because it has no default value.

`limit` is optional because it has a default value.

## Runtime Flow

During an agent run:

```text
The model requests a tool call.
Sophons finds the tool by name.
Sophons calls the Python function.
Sophons sends the tool result back to the model.
The model continues or returns a final answer.
```

The model decides when to request a tool. Sophons executes the tool.

## Required Arguments

Arguments without defaults are required.

```python
@tool
def get_customer(customer_id: str) -> dict:
    """Get a customer by ID."""
    return {"id": customer_id}
```

Sophons marks `customer_id` as a required string argument.

## Optional Arguments

Arguments with defaults are optional.

```python
@tool
def search_docs(query: str, limit: int = 5) -> dict:
    """Search documents."""
    return {
        "query": query,
        "limit": limit,
        "documents": [],
    }
```

Sophons marks `query` as required and `limit` as optional with default `5`.

## Type Hints

Sophons uses type hints to build the argument schema shown to the model.

| Python type | Schema type |
|---|---|
| `str` | `string` |
| `int` | `integer` |
| `float` | `number` |
| `bool` | `boolean` |
| `dict` | `object` |
| `list[T]` | `array` |

For example:

```python
@tool
def create_ticket(title: str, priority: int, urgent: bool = False) -> dict:
    """Create a support ticket."""
    return {
        "title": title,
        "priority": priority,
        "urgent": urgent,
    }
```

Sophons infers:

```text
title: required string
priority: required integer
urgent: optional boolean with default false
```

## Return Values

Tools return data back to the agent.

If a tool returns a dictionary, Sophons preserves it.

```python
@tool
def search_docs(query: str) -> dict:
    """Search documents."""
    return {
        "documents": [
            {"id": "doc_1", "content": "Sophons supports tools."}
        ]
    }
```

If a tool returns a simple value, Sophons wraps it in a dictionary.

```python
@tool
def count_letters(text: str, letter: str) -> int:
    """Count how many times a letter appears in text."""
    return text.count(letter)
```

The tool result becomes:

```python
{"result": 3}
```

## Errors

Raise normal Python exceptions when a tool cannot complete.

```python
@tool
def divide(a: float, b: float) -> float:
    """Divide one number by another."""
    if b == 0:
        raise ValueError("b must not be zero")
    return a / b
```

The agent loop turns tool exceptions into error tool results. The model can then
recover, try another approach, or explain what went wrong.

## Passing Tools To Agents

Pass tools to `Agent` with the `tools` argument.

```python
agent = Agent(
    model=my_model,
    tools=[search_docs, count_letters],
)
```

The agent can only call tools that you pass in.

## Class-Based Tools

Use class methods when tools need shared state, clients, or configuration.

```python
from sophons.tools import tool


class DocumentTools:
    def __init__(self, retriever):
        self.retriever = retriever

    @tool
    def search(self, query: str, limit: int = 5) -> dict:
        """Search the document index."""
        documents = self.retriever.retrieve(query, limit=limit)
        return {
            "documents": [document.to_dict() for document in documents],
        }


document_tools = DocumentTools(retriever)

agent = Agent(
    model=my_model,
    tools=[document_tools.search],
)
```

This pattern is useful for database clients, API clients, retrievers, memory
stores, and other long-lived resources.

## Custom Tool Objects

Most tools should use `@tool`.

For lower-level control, implement the `Tool` protocol directly.

```python
class SearchDocsTool:
    name = "search_docs"
    description = "Search documents."
    args_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "limit": {"type": "integer", "default": 5},
        },
        "required": ["query"],
    }

    def call(self, args: dict) -> dict:
        query = args["query"]
        limit = args.get("limit", 5)
        return {
            "query": query,
            "limit": limit,
            "documents": [],
        }
```

Anything with `name`, `description`, `args_schema`, and `call(args)` can be used
as a Sophons tool.

## Current Limits

The first `@tool` implementation is intentionally small.

Today it supports:

- function names
- docstring descriptions
- type hints
- default values
- dictionary return values
- simple return values
- sync functions
- class methods

It does not yet infer parameter descriptions from docstring `Args:` blocks.

For example, this description is not used yet:

```python
@tool
def search_docs(query: str) -> dict:
    """Search documents.

    Args:
        query: The search query.
    """
    ...
```

Planned improvements:

- parameter descriptions from docstrings
- custom tool names
- custom descriptions
- custom input schemas
- async tools
- explicit tool context for state and sessions
- richer tool result content blocks

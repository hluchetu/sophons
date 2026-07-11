---
title: Agents
description: The sophons.agents module — async agent loop, conversation management, retry, hooks, and session persistence.
---

# sophons.agents

The agents module is the core of the SDK. It provides everything needed to run
a production agent loop: conversation management, tool execution, retry, lifecycle
hooks, and session persistence.

---

## Architecture

```
Agent
 └── AgentLoop
      ├── ConversationManager   — prepares context window before each model call
      ├── RetryStrategy         — retries failed model calls with backoff
      ├── ToolExecutor          — calls tools and returns results to the loop
      ├── HookManager           — fires lifecycle events (start, model call, tool call, end)
      └── SessionManager        — saves and loads conversation history across runs
```

---

## Agent

The public entry point. Everything is wired up inside; you pass the pieces you want.

```python
from sophons.agents import Agent

agent = Agent(
    model=my_model,                         # required: AsyncChatModel
    tools=[search_tool, calculator_tool],   # optional
    system_prompt="You are a helpful assistant.",
    hooks=my_hooks,                         # optional: AgentHooks
    session_manager=FileSessionManager(".sessions"),  # optional
    retry_strategy=RetryStrategy(max_attempts=3),     # optional
    limits=RunLimits(max_turns=10),                   # optional
)

# Async (primary)
result = await agent.run("What is 2 + 2?")

# Sync (convenience wrapper)
result = agent.run_sync("What is 2 + 2?")

print(result.output)       # final assistant message
print(result.turns)        # number of turns taken
print(result.messages)     # full conversation history
```

---

## Conversation management

`ConversationManager` decides which messages go into the model call on each turn.
The default implementation passes all messages. Swap it out to add truncation,
summarisation, or RAG-based injection.

```python
from sophons.agents import SlidingWindowConversationManager

agent = Agent(
    model=my_model,
    conversation_manager=SlidingWindowConversationManager(max_messages=20),
)
```

---

## Tools

Tools are callables — sync or async — that the agent can invoke. Annotate with
`@tool` to register metadata, or pass plain callables.

```python
from sophons.agents import tool

@tool(name="search", description="Search the web for a query.")
async def search(query: str) -> str:
    ...

agent = Agent(model=my_model, tools=[search])
```

The agent loop automatically detects tool calls in the model response, executes
them, and feeds results back as `tool` messages.

---

## Retry

`RetryStrategy` wraps model calls. It retries on transient failures and supports
both sync and async retry policies.

```python
from sophons.agents import RetryStrategy, RetryConfig

agent = Agent(
    model=my_model,
    retry_strategy=RetryStrategy(
        RetryConfig(max_attempts=3, backoff_base=1.0)
    ),
)
```

---

## Hooks

Hooks let you observe and react to the agent lifecycle without modifying the loop.
Use them for logging, tracing, or injecting memory into the context.

```python
from sophons.agents import AgentHooks, RunContext

class MyHooks(AgentHooks):
    async def on_model_call(self, context: RunContext) -> None:
        print(f"Calling model, turn {context.turn}")

    async def on_tool_call(self, context: RunContext, tool_name: str) -> None:
        print(f"Calling tool: {tool_name}")
```

---

## Session persistence

`SessionManager` saves and loads conversation history so the agent can resume
across process restarts.

```python
from sophons.agents import FileSessionManager

session_manager = FileSessionManager(directory=".sessions")
agent = Agent(model=my_model, session_manager=session_manager)

# Pass a session_id to continue an existing conversation
result = await agent.run("Continue from where we left off", session_id="user-123")
```

Built-in implementations:

| Class | Backend |
|---|---|
| `InMemorySessionManager` | Dict — no persistence, suitable for tests |
| `FileSessionManager` | JSON files on disk |

---

## Run limits

```python
from sophons.agents import RunLimits

agent = Agent(
    model=my_model,
    limits=RunLimits(max_turns=10, max_tokens=4096),
)
```

---

## AgentResult

Every `run()` call returns an `AgentResult`:

```python
result.output      # str — final assistant message
result.messages    # list[Message] — full conversation
result.turns       # int — number of turns taken
result.session_id  # str | None
```

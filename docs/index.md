---
title: Sophons
description: A production-ready Python SDK for building agents — async-first, modular, and framework-agnostic.
---

# Sophons

Sophons is a Python SDK for building production-grade AI agents. It handles the
hard parts — the agent loop, conversation management, retry, tool execution,
long-term memory, and retrieval — so you focus on what your agent actually does.

---

## Design principles

- **Async-first.** Every entry point is `async def`. Sync wrappers are provided for convenience.
- **Protocol-based.** Sophons defines contracts. You bring the backends — any LLM, any vector store, any storage engine.
- **No hidden magic.** State is explicit, side effects are visible, and every layer is independently testable.
- **Modular.** Use only what you need. Agents, memory, and RAG are independent — none requires the others.

---

## Modules

| Module | What it does |
|---|---|
| [`sophons.agents`](/docs/sophons/agents) | Agent loop, conversation management, retry, hooks, session persistence |
| [`sophons.tools`](/docs/sophons/tools) | Decorate Python functions as agent-callable tools |
| [`sophons.memory`](/docs/sophons/memory) | Long-term memory storage, retrieval, extraction, and reflection |
| [`sophons.rag`](/docs/sophons/rag) | Document splitting, retrieval, compression, and RAG pipeline |
| `sophons.models` | `ChatModel` and `AsyncChatModel` protocols — bring your own model adapter |
| `sophons.documents` | Provider-neutral `Document` type used across modules |
| `sophons.retrieval` | `BM25Retriever` — built-in lexical retriever, no external deps |

---

## Quick start

```python
from sophons.agents import Agent
from sophons.memory import MemoryManager, MemoryStore, MemoryStoreConfig, InMemoryStorage, LexicalRetriever

# Wire up memory
store = MemoryStore(storage=InMemoryStorage(), retrievers=[LexicalRetriever()])
memory = MemoryManager(
    stores=[MemoryStoreConfig(name="main", description="main memory", store=store)]
)

# Create an agent (bring your own model adapter)
agent = Agent(model=my_model)

# Run
result = await agent.run("What is the capital of France?")
print(result.output)
```

---

## Installation

```bash
pip install sophons
```

For local development against the source:

```bash
pip install -e /path/to/sophons
```

---

## What's next

- [Agents](/docs/sophons/agents) — how the agent loop works
- [Tools](/docs/sophons/tools) — how to define functions agents can call
- [Memory](/docs/sophons/memory) — long-term memory from scratch
- [RAG](/docs/sophons/rag) — document splitting and retrieval

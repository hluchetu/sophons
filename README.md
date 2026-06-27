# Sophons

Sophons is a Python SDK for agent memory, retrieval, and RAG.

The goal is to provide small, reusable building blocks that can be used across different agent SDKs. Sophons should not be tied to one framework, one model provider, or one memory implementation.

## What Sophons Provides

- Short-term conversation memory
- Long-term memory
- Generic retrievers for RAG, memory, and code/docs search
- RAG workflows built on retrievers
- Agent-callable tools
- Integrations with agent SDKs

## Package Shape

```text
sophons/
  retrieval/       generic retrievers: vector, BM25, hybrid, rerank
  rag/             RAG workflows built on retrievers
  memory/          short-term and long-term memory
  tools/           expose retrievers and memory as agent tools
  integrations/    adapters for agent SDKs
```

## Retriever Pattern

Retrievers follow one simple contract:

```text
query -> documents
```

That means the same retriever can be used in different places:

- A RAG app can retrieve documents before answering.
- A memory system can retrieve relevant user memories.
- An agent can expose a retriever as a tool.
- A framework adapter can wrap the retriever for OpenAI, Strands, LangChain-style APIs, or another SDK.

The first retriever layer will focus on the common foundation:

- Vector retrieval
- BM25 lexical retrieval
- Hybrid retrieval
- Parent document retrieval

More advanced retrievers can wrap or combine those foundations:

- Multi-query retrieval
- Contextual compression
- Self-query retrieval
- Reranking
- Routing
- Ensemble retrieval

## Status

Sophons is at the SDK skeleton stage. The package layout is being shaped first so the core abstractions stay clean before implementation is moved in.

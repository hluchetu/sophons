# Sophons

Sophons is a Python SDK for agent memory, retrieval, and RAG.

The goal is to provide small, reusable building blocks that can be used across different agent SDKs. Sophons should not be tied to one framework, one model provider, or one memory implementation.

## Status

Sophons is early-stage. The current repo contains the package skeleton, shared data types, and base interfaces for the first SDK layers.

Implemented so far:

- `Document`
- `Message`
- `Retriever` / `AsyncRetriever`
- `Loader` / `AsyncLoader`
- `Splitter`
- `Tool` / `AsyncTool`

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
  documents/       shared document schema
  models/          messages and model interfaces
  loaders/         source -> documents
  splitters/       documents -> chunks
  retrieval/       generic retrievers: vector, BM25, hybrid, rerank
  rag/             RAG workflows built on retrievers
  memory/          short-term and long-term memory
  tools/           expose retrievers and memory as agent tools
  integrations/    adapters for agent SDKs
```

## Install For Local Development

```bash
pip install -e .
```

Run tests:

```bash
python -m pytest
```

## Core Types

```python
from sophons import Document, Message

document = Document(
    id="auth.md#chunk_1",
    content="Token refresh happens every 60 minutes.",
    metadata={"source": "auth.md"},
)

message = Message(
    role="user",
    content="How does token refresh work?",
)
```

`Document` is for knowledge/context.

`Message` is for model and agent conversation.

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

## Loader And Splitter Pattern

Loaders ingest external sources:

```text
source -> documents
```

Splitters prepare documents for indexing and retrieval:

```text
documents -> chunks
```

Together:

```text
source -> Loader -> Documents -> Splitter -> Chunks -> Retriever
```

## Tool Pattern

Tools follow a simple agent-facing contract:

```text
structured args -> structured result
```

Tools are how agents will call Sophons capabilities such as memory search, document retrieval, or RAG context retrieval.

## Roadmap

Next small implementation steps:

1. Text loading and recursive splitting
2. BM25 lexical retrieval
3. DeepSeek chat integration
4. Basic BM25-based RAG pipeline
5. Hugging Face embeddings and vector retrieval

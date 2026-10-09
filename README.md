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
- `BM25Retriever`, `SemanticRetriever`, `HybridRetriever`, `MultiQueryRetriever`

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
uv sync --group dev
```

Run tests:

```bash
uv run python -m pytest
```

Install an optional embedding provider when needed:

```bash
uv sync --group dev --extra openai
# or
uv sync --group dev --extra sentence-transformers
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

## Embeddings

`sophons.embeddings` is the stable public embedding API. Core protocols are
always available, while provider integrations and their SDKs are loaded only
when requested.

```python
from sophons.embeddings import EmbeddingModel, OpenAIEmbeddings

embedder: EmbeddingModel = OpenAIEmbeddings(
    api_key="...",
    model="text-embedding-3-small",
)

query_vector = embedder.embed_query("How does token refresh work?")
document_vectors = embedder.embed_documents([
    "Token refresh happens every 60 minutes.",
    "Sessions expire after 24 hours.",
])
```

For asynchronous applications:

```python
from sophons.embeddings import AsyncOpenAIEmbeddings

embedder = AsyncOpenAIEmbeddings(api_key="...")
query_vector = await embedder.embed_query("How does token refresh work?")
```

Applications should import from `sophons.embeddings`, not from internal
`sophons.integrations` modules. Provider packages remain optional: importing
the public namespace does not import OpenAI or Sentence Transformers.

Sentence Transformers can return L2-normalized vectors when the consuming
application uses cosine similarity or dot product over unit vectors:

```python
from sophons.embeddings import SentenceTransformerEmbeddings

embedder = SentenceTransformerEmbeddings(
    model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    normalize=True,
)
```

Normalization is explicit and disabled by default so Sophons does not impose a
similarity policy on every consumer.

## Models

`sophons.models` is also the stable public API for chat-model contracts,
messages, settings, and provider implementations. Provider SDKs are loaded only
when their model is constructed.

Install DeepSeek support:

```bash
uv sync --group dev --extra deepseek
```

Then create the model through the public API:

```python
from sophons.models import DeepSeekModel, Message

model = DeepSeekModel(
    model="deepseek-chat",
    api_key="...",
)

response = model.invoke([
    Message(role="user", content="Explain Article 43."),
])
```

The provider-independent `ChatModel` protocol remains available from the same
namespace:

```python
from sophons.models import ChatModel, DeepSeekModel

model: ChatModel = DeepSeekModel(
    model="deepseek-chat",
    api_key="...",
)
```

Applications should prefer `sophons.models` over internal
`sophons.integrations.models` imports.

## Vector Stores

`sophons.stores` is the stable public API for vector-store contracts and
implementations. The in-memory store has no optional dependency:

```python
from sophons.stores import InMemoryVectorStore, VectorStore

store: VectorStore = InMemoryVectorStore()
```

Chroma remains optional and is loaded only when constructed:

```bash
uv sync --extra chroma
```

```python
from sophons.stores import ChromaVectorStore

store = ChromaVectorStore(
    collection="katiba",
    path="./katiba_index",
)
```

Applications should prefer `sophons.stores` over internal
`sophons.integrations.vector_stores` imports.

For exact cosine search over an existing NumPy matrix:

```bash
uv sync --extra numpy
```

```python
from sophons.stores import NumPyVectorStore

store = NumPyVectorStore()
store.add_matrix(documents, vectors)
results = store.search(query_vector, limit=10)
```

`NumPyVectorStore` keeps vectors as a `float32` matrix and performs bulk cosine
search without converting the matrix to nested Python lists.

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

The first retriever layer includes the common foundation:

- Vector retrieval
- BM25 lexical retrieval
- Hybrid retrieval
- Multi-query retrieval

More advanced retrievers can wrap or combine those foundations:

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

To read a file as structured elements instead of flat text, parse it:

```python
from sophons.parsers import Blob, default_registry
from sophons.splitters import StructureAwareSplitter

parsed = default_registry().parse(Blob.from_path("contract.docx"))
for element in parsed.leaves():
    print(element.kind, parsed.text_of(element)[:60])
chunks = StructureAwareSplitter(max_chunk_size=1000).split_parsed(parsed)
```

A parser takes bytes and returns one `ParsedDocument`: the extracted text plus
typed elements (headings, paragraphs, list items, tables with rows and cells)
recorded as offsets into that text. Built-in parsers cover plain text, Markdown,
PDF text layers (`sophons[pdf]`) and Word. Chunks keep their offsets, pages and
heading path, so a quotation can be traced to its place in the source.

For structure-aware PDFs, install `sophons[pdf-structure]` and use:

```python
from sophons.loaders import PDFStructureLoader
from sophons.splitters import StructureAwareSplitter

document = PDFStructureLoader("contract.pdf").load()[0]
chunks = StructureAwareSplitter(max_chunk_size=1000).split(document)
```

The loader converts PDF layout to Markdown with PyMuPDF4LLM. The splitter
preserves detected sections and blocks, with source page metadata. See
[the walkthrough](docs/structure_chunking.md) for inspecting intermediate
structure, OCR options, and limitations.

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

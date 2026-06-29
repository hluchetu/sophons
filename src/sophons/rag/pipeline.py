from __future__ import annotations

from uuid import uuid4

from sophons.documents.base import Document
from sophons.models.chat import AsyncChatModel
from sophons.models.messages import Message
from sophons.rag.compressors.interface import DocumentCompressor
from sophons.rag.splitters.interface import TextSplitter
from sophons.rag.splitters.recursive import RecursiveTextSplitter
from sophons.retrieval.bm25 import BM25Retriever


_GENERATION_PROMPT = """\
Answer the question using only the context provided below. \
If the context does not contain enough information, say so.

Context:
{context}

Question: {question}"""


class RAGPipeline:
    """
    Thin orchestrator for a Retrieval-Augmented Generation pipeline.

    Wires together a text splitter, a retriever, an optional compressor
    (reranker), and an optional generator (LLM) into a single object with
    three entry points:

    - ``index(document)`` — split and index a document.
    - ``retrieve(query)`` — fetch the most relevant chunks.
    - ``query(question)`` — retrieve + generate an answer.

    Unlike LangChain's LCEL approach, ``RAGPipeline`` is a plain class — it
    is simpler to use without a chain DSL and easy to swap individual stages.

    By default the pipeline uses a ``BM25Retriever`` from
    ``sophons.retrieval.bm25``.  For semantic or hybrid retrieval pass your
    own retriever via the ``retriever`` parameter.

    Args:
        splitter:   Splits documents into chunks.  Defaults to
                    ``RecursiveTextSplitter``.
        retriever:  Retrieves relevant chunks by query.  Defaults to the
                    built-in ``BM25Retriever``.
        compressor: Optional reranker / filter applied after retrieval.
        model:      Optional async chat model used for answer generation in
                    ``query()``.  Required only if you call ``query()``.
        top_k:      Number of chunks to return from ``retrieve()``.

    Usage::

        pipeline = RAGPipeline(model=my_model)
        pipeline.index(Document(content=long_text, id="doc-1"))
        chunks = await pipeline.retrieve("what is the capital of France?")
        answer = await pipeline.query("what is the capital of France?")
    """

    def __init__(
        self,
        splitter: TextSplitter | None = None,
        retriever: BM25Retriever | None = None,
        compressor: DocumentCompressor | None = None,
        model: AsyncChatModel | None = None,
        top_k: int = 5,
    ) -> None:
        self._splitter = splitter or RecursiveTextSplitter()
        self._retriever = retriever or BM25Retriever([])
        self._compressor = compressor
        self._model = model
        self._top_k = top_k
        # BM25Retriever needs to be rebuilt when new documents are indexed
        self._indexed_docs: list[Document] = []

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index(self, document: Document) -> int:
        """
        Split ``document`` and add the chunks to the retriever index.

        Returns the number of chunks indexed.
        """
        chunks = self._splitter.split(document)
        if not chunks:
            return 0
        self._indexed_docs.extend(chunks)
        # Rebuild the BM25 index with all documents so far
        if isinstance(self._retriever, BM25Retriever):
            self._retriever = BM25Retriever(self._indexed_docs)
        return len(chunks)

    async def index_async(self, document: Document) -> int:
        """Async variant of ``index()`` — required when using ``ContextualTextSplitter``."""
        from sophons.rag.splitters.contextual import ContextualTextSplitter

        if isinstance(self._splitter, ContextualTextSplitter):
            chunks = await self._splitter.split_async(document)
        else:
            chunks = self._splitter.split(document)

        if not chunks:
            return 0
        self._indexed_docs.extend(chunks)
        if isinstance(self._retriever, BM25Retriever):
            self._retriever = BM25Retriever(self._indexed_docs)
        return len(chunks)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    async def retrieve(self, query: str) -> list[Document]:
        """
        Retrieve the most relevant chunks for ``query``.

        Applies the compressor if one is configured.
        """
        results = self._retriever.retrieve(query, limit=self._top_k)

        if self._compressor is not None and results:
            results = await self._compressor.compress(results, query)

        return results[: self._top_k]

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    async def query(self, question: str) -> str:
        """
        Retrieve relevant chunks and generate an answer with the LLM.

        Raises ``ValueError`` when no model is configured.
        """
        if self._model is None:
            raise ValueError(
                "No model configured. Pass a model to RAGPipeline.__init__() "
                "to use query()."
            )

        chunks = await self.retrieve(question)
        if not chunks:
            context = "No relevant context found."
        else:
            context = "\n\n".join(c.content for c in chunks)

        prompt = _GENERATION_PROMPT.format(context=context, question=question)
        response = await self._model.invoke(
            [Message(role="user", content=prompt)]
        )
        return response.content

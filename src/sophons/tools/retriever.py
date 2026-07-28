from __future__ import annotations

from dataclasses import dataclass, field

from sophons.retrieval.base import Retriever
from sophons.tools.base import ToolArgs, ToolResult, ToolSchema


@dataclass(frozen=True, slots=True)
class RetrieverTool:
    """
    Adapts a retriever into an agent tool.

    The agent sees a normal tool: a name, description, argument schema,
    and call method. Internally, the tool runs the retriever and returns
    the top passages as text.

    Use this when retrieval should be an action the agent can choose,
    rather than a pipeline step that always runs.

    Args:
        name:        Tool name the model uses to invoke it.
        description: What this knowledge base contains. Be specific —
                     the model reads this to decide when to call the tool.
        retriever:   Any ``Retriever`` — ``BM25Retriever``, ``SemanticRetriever``,
                     or your own.
        top_k:       Number of documents to return per search. Defaults to 5.

    Usage::

        tool = RetrieverTool(
            name="search_docs",
            description="Search company policy documents.",
            retriever=SemanticRetriever(embedder=..., vector_store=...),
        )
        agent = Agent(model=my_model, tools=[tool])
    """

    name: str
    description: str
    retriever: Retriever
    top_k: int = field(default=5)

    @property
    def args_schema(self) -> ToolSchema:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to look up in the knowledge base.",
                }
            },
            "required": ["query"],
        }

    def call(self, args: ToolArgs) -> ToolResult:
        docs = self.retriever.retrieve(args["query"], limit=self.top_k)
        if not docs:
            return {"result": "No relevant documents found for that query."}
        passages = "\n\n".join(
            f"[{i + 1}] {doc.content}" for i, doc in enumerate(docs)
        )
        return {"result": passages}

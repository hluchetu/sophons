from __future__ import annotations

from typing import Any

from sophons.documents import Document


class ChromaVectorStore:
    """
    Persistent vector store backed by ChromaDB.

    Suitable for larger corpora and production use. Vectors are persisted to
    disk and survive process restarts.

    Requires ``chromadb`` to be installed::

        pip install chromadb

    Args:
        collection:  Name of the ChromaDB collection.
        path:        Directory to persist data. Defaults to ``./chroma_db``.
                     Pass ``None`` to use an in-memory ChromaDB client (useful
                     for testing without disk writes).

    Usage::

        store = ChromaVectorStore(collection="my-docs", path="./chroma_db")
        store.add(documents, vectors)
        results = store.search(query_vector, limit=5)
    """

    def __init__(
        self,
        collection: str,
        path: str | None = "./chroma_db",
    ) -> None:
        try:
            import chromadb
        except ImportError as exc:
            raise ImportError(
                "chromadb is required for ChromaVectorStore. "
                "Install it with: pip install chromadb"
            ) from exc

        if path is None:
            self._client = chromadb.Client()
        else:
            self._client = chromadb.PersistentClient(path=path)

        self._collection = self._client.get_or_create_collection(
            name=collection,
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, documents: list[Document], vectors: list[list[float]]) -> None:
        """Store documents alongside their embedding vectors."""
        if not documents:
            return

        ids = [doc.id or str(i) for i, doc in enumerate(documents)]
        self._collection.upsert(
            ids=ids,
            embeddings=vectors,
            documents=[doc.content for doc in documents],
            metadatas=[dict(doc.metadata) for doc in documents],
        )

    def search(self, vector: list[float], *, limit: int = 10) -> list[Document]:
        """Return the ``limit`` most similar documents by cosine similarity."""
        results = self._collection.query(
            query_embeddings=[vector],
            n_results=min(limit, self._collection.count() or 1),
            include=["documents", "metadatas", "distances"],
        )

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        ids = results.get("ids", [[]])[0]

        return [
            Document(
                content=doc,
                metadata=meta,
                id=doc_id,
                score=1.0 - dist,
            )
            for doc, meta, dist, doc_id in zip(documents, metadatas, distances, ids)
        ]

    def delete(self, ids: list[str]) -> None:
        """Remove documents by their IDs."""
        self._collection.delete(ids=ids)

    def count(self) -> int:
        """Return the number of documents in the collection."""
        return self._collection.count()

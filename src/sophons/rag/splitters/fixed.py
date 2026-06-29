from __future__ import annotations

from sophons.documents.base import Document
from sophons.rag.splitters.interface import TextSplitter


class FixedSizeTextSplitter:
    """
    Splits text into fixed-size character chunks with optional overlap.

    The simplest splitting strategy — divides text into chunks of exactly
    ``chunk_size`` characters (or fewer for the final chunk), with each
    successive chunk starting ``overlap`` characters before the previous
    chunk ended.

    Args:
        chunk_size: Maximum number of characters per chunk.
        overlap:    Number of characters to overlap between consecutive chunks.
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")
        if overlap < 0 or overlap >= chunk_size:
            raise ValueError("overlap must be >= 0 and < chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split(self, document: Document) -> list[Document]:
        text = document.content
        if not text.strip():
            return []

        chunks: list[str] = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end >= len(text):
                break
            start = end - self.overlap

        return [
            Document(
                content=chunk,
                metadata={**document.metadata, "chunk_index": i},
                id=f"{document.id}_chunk_{i}" if document.id else None,
            )
            for i, chunk in enumerate(chunks)
        ]

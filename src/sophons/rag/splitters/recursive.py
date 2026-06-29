from __future__ import annotations

from sophons.documents.base import Document


class RecursiveTextSplitter:
    """
    Splits text by trying a priority list of separators recursively.

    Mirrors LangChain's ``RecursiveCharacterTextSplitter`` — the most
    commonly used splitter in production RAG pipelines.  It tries separators
    in order (paragraphs → newlines → sentences → words → characters) and
    recurses into any piece that still exceeds ``chunk_size``.

    The result is semantically coherent chunks: the splitter prefers to break
    at paragraph boundaries, falls back to sentence boundaries, and only
    splits mid-word as a last resort.

    Args:
        chunk_size:  Maximum characters per chunk.
        overlap:     Characters to overlap between consecutive chunks.
        separators:  Ordered list of separators to try.  Defaults to
                     ``["\\n\\n", "\\n", ". ", " ", ""]``.
    """

    _DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(
        self,
        chunk_size: int = 500,
        overlap: int = 50,
        separators: list[str] | None = None,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.separators = separators or self._DEFAULT_SEPARATORS

    def split(self, document: Document) -> list[Document]:
        text = document.content
        if not text.strip():
            return []

        raw_chunks = self._split(text, self.separators)
        return [
            Document(
                content=chunk,
                metadata={**document.metadata, "chunk_index": i},
                id=f"{document.id}_chunk_{i}" if document.id else None,
            )
            for i, chunk in enumerate(raw_chunks)
        ]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _split(self, text: str, separators: list[str]) -> list[str]:
        separator = separators[0]
        remaining = separators[1:]

        chunks: list[str] = []
        current = ""
        pieces = text.split(separator) if separator else list(text)

        for piece in pieces:
            candidate = current + separator + piece if current else piece

            if len(piece) > self.chunk_size and remaining:
                # This piece alone is too big — recurse with the next separator
                if current:
                    chunks.extend(self._finalise(current))
                    current = ""
                chunks.extend(self._split(piece, remaining))

            elif len(candidate) <= self.chunk_size:
                current = candidate

            else:
                if current:
                    chunks.extend(self._finalise(current))
                current = piece

        if current:
            chunks.extend(self._finalise(current))

        return [c for c in chunks if c.strip()]

    def _finalise(self, text: str) -> list[str]:
        """Return ``text`` as-is if it fits; hard-split it if it is too large."""
        if len(text) <= self.chunk_size:
            return [text.strip()]
        # Last-resort hard split with overlap
        result: list[str] = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end].strip()
            if chunk:
                result.append(chunk)
            if end >= len(text):
                break
            start = end - self.overlap
        return result

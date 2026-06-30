from __future__ import annotations

from collections.abc import Iterable

from sophons.documents import Document


class RecursiveCharacterSplitter:
    """Split documents by trying natural separators before smaller units."""

    def __init__(
        self,
        *,
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
        separators: list[str] | None = None,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0.")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be greater than or equal to 0.")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size.")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]

    def split_document(self, document: Document) -> list[Document]:
        chunks = self.split_text(document.content)
        return [
            Document(
                id=self._chunk_id(document, index),
                content=chunk,
                metadata={
                    **document.metadata,
                    "parent_id": document.id,
                    "chunk_index": index,
                },
            )
            for index, chunk in enumerate(chunks)
        ]

    def split_documents(self, documents: Iterable[Document]) -> list[Document]:
        chunks: list[Document] = []
        for document in documents:
            chunks.extend(self.split_document(document))
        return chunks

    def split_text(self, text: str) -> list[str]:
        stripped = text.strip()
        if not stripped:
            return []

        pieces = self._split_recursive(stripped, self.separators)
        return self._merge_pieces(pieces)

    def _split_recursive(self, text: str, separators: list[str]) -> list[str]:
        if len(text) <= self.chunk_size:
            return [text]

        separator = separators[0] if separators else ""
        remaining_separators = separators[1:] if separators else []

        if separator and separator in text:
            raw_pieces = self._split_keep_separator(text, separator)
        elif separator == "":
            raw_pieces = list(text)
        else:
            return self._split_recursive(text, remaining_separators)

        pieces: list[str] = []
        for piece in raw_pieces:
            piece = piece.strip()
            if not piece:
                continue
            if len(piece) <= self.chunk_size:
                pieces.append(piece)
            else:
                pieces.extend(self._split_recursive(piece, remaining_separators))

        return pieces

    def _merge_pieces(self, pieces: list[str]) -> list[str]:
        chunks: list[str] = []
        current = ""

        for piece in pieces:
            candidate = self._join(current, piece)
            if len(candidate) <= self.chunk_size:
                current = candidate
                continue

            if current:
                chunks.append(current)
                current = self._join(self._overlap_text(current), piece)
            else:
                current = piece

        if current:
            chunks.append(current)

        return chunks

    def _overlap_text(self, text: str) -> str:
        if not self.chunk_overlap:
            return ""
        return text[-self.chunk_overlap :].strip()

    @staticmethod
    def _join(left: str, right: str) -> str:
        if not left:
            return right.strip()
        if not right:
            return left.strip()
        return f"{left.rstrip()} {right.lstrip()}".strip()

    @staticmethod
    def _split_keep_separator(text: str, separator: str) -> list[str]:
        parts = text.split(separator)
        pieces: list[str] = []
        for index, part in enumerate(parts):
            if index < len(parts) - 1:
                pieces.append(part + separator)
            else:
                pieces.append(part)
        return pieces

    @staticmethod
    def _chunk_id(document: Document, index: int) -> str | None:
        if document.id is None:
            return None
        return f"{document.id}#chunk_{index}"

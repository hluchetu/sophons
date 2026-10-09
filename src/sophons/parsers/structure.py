from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

from sophons.documents import Document


@dataclass(frozen=True)
class DocumentBlock:
    """A structural unit with its enclosing headings and source line range."""

    content: str
    kind: str
    heading_path: tuple[str, ...] = ()
    start_line: int = 1
    end_line: int = 1


class StructureParser(Protocol):
    def parse(self, document: Document) -> list[DocumentBlock]: ...


class TextStructureParser:
    """Recognize text structure without changing the source text.

    Numbered headings are opt-in heuristics, not a PDF layout detector.
    Lines beginning with dotted numbers (7. or 7.2) become boundaries.
    """

    def __init__(self, *, numbered_sections: bool = False) -> None:
        self.numbered_sections = numbered_sections

    def parse(self, document: Document) -> list[DocumentBlock]:
        lines = document.content.splitlines()
        blocks: list[DocumentBlock] = []
        headings: list[tuple[int, str]] = []
        index = 0
        while index < len(lines):
            line = lines[index]
            if not line.strip():
                index += 1
                continue
            heading = self._heading(line)
            if heading:
                level, title = heading
                headings = [(depth, name) for depth, name in headings if depth < level]
                headings.append((level, title))
            start = index
            kind = 'heading' if heading else self._kind(line)
            index += 1
            if kind == 'code':
                fence = re.match(r'^\s*(`{3,}|~{3,})', line).group(1)
                while index < len(lines):
                    closing = re.fullmatch(r'\s*' + re.escape(fence[0]) + '{' + str(len(fence)) + r',}\s*', lines[index])
                    index += 1
                    if closing:
                        break
            elif kind != 'heading':
                while index < len(lines) and lines[index].strip():
                    if self._heading(lines[index]) or self._kind(lines[index]) != kind:
                        break
                    index += 1
            blocks.append(DocumentBlock(
                content='\n'.join(lines[start:index]), kind=kind,
                heading_path=tuple(name for _, name in headings),
                start_line=start + 1, end_line=index,
            ))
        return blocks

    def _heading(self, line: str) -> tuple[int, str] | None:
        match = re.match(r'^\s{0,3}(#{1,6})\s+(.+?)\s*$', line)
        if match:
            return len(match[1]), match[2]
        if self.numbered_sections:
            match = re.match(r'^\s*(\d+(?:\.\d+)*)(?:\.(?=\s)|(?<=\d)(?=\s))\s+(.+)$', line)
            if match:
                return match[1].count('.') + 1, f'{match[1]} {match[2]}'
        return None

    @staticmethod
    def _kind(line: str) -> str:
        if re.match(r'^\s*(`{3,}|~{3,})', line):
            return 'code'
        if re.match(r'^\s*\|.*\|\s*$', line):
            return 'table'
        if re.match(r'^\s*(?:[-+*]|\d+[.)])\s+', line):
            return 'list'
        return 'paragraph'

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import replace
from typing import Protocol, runtime_checkable

from sophons.parsers.elements import Element, ParsedDocument

_PAGE_NUMBER = re.compile(r"^\W*(?:page\s+)?\d{1,4}(?:\s*(?:of|/)\s*\d{1,4})?\W*$", re.IGNORECASE)
# "7." or "7)" or a dotted number such as "7.2"; a bare number is too often a year or amount.
_CLAUSE = re.compile(r"^[ \t]*(\d{1,3}(?:\.\d{1,3})+\.?|\d{1,3}[.)])[ \t]+\S", re.MULTILINE)
_ENDS_SENTENCE = re.compile(r"[.!?:;][\"'”’)\]]*\s*$")
_CONTENT = ("paragraph", "list_item")


@runtime_checkable
class Cleaner(Protocol):
    """Cleaner contract: a parsed document in, a still-valid parsed document out.

    Cleaners never change ``ParsedDocument.text``. They reclassify, split or link
    elements, so every offset recorded before cleaning stays true afterwards.
    """

    name: str

    def clean(self, parsed: ParsedDocument) -> ParsedDocument:
        ...


def clean(parsed: ParsedDocument, cleaners: Iterable[Cleaner] | None = None) -> ParsedDocument:
    """Apply cleaners in order, validating after each one."""

    for cleaner in default_cleaners() if cleaners is None else cleaners:
        parsed = cleaner.clean(parsed).validate()
    return parsed


def default_cleaners() -> list[Cleaner]:
    """Page furniture first, so later steps see only real content."""

    return [MarkPageFurniture(), SplitNumberedClauses(), LinkAcrossPages()]


def _trimmed(text: str, start: int, end: int) -> tuple[int, int]:
    """Shrink a span so it holds no leading or trailing whitespace."""

    body = text[start:end]
    return start + len(body) - len(body.lstrip()), start + len(body.rstrip())


def _rebuild(parsed: ParsedDocument, replacements: dict[str, list[Element]]) -> ParsedDocument:
    elements: list[Element] = []
    for element in parsed.elements:
        elements.extend(replacements.get(element.id, [element]))
    return replace(parsed, elements=tuple(elements))


class MarkPageFurniture:
    """Reclassify page numbers and running headers or footers.

    Looks only at the first and last line of each page. A line is furniture if
    it is a bare page number, or if the same line (digits aside) sits in that
    position on most pages. The text is kept; the element becomes a ``header`` or
    ``footer`` so chunking and rendering can leave it out.
    """

    name = "mark-page-furniture"

    def __init__(self, *, repeat_share: float = 0.6, min_pages: int = 3) -> None:
        self.repeat_share = repeat_share
        self.min_pages = min_pages

    def clean(self, parsed: ParsedDocument) -> ParsedDocument:
        pages = [element for element in parsed.elements if element.kind == "page"]
        if not pages:
            return parsed
        edges: dict[str, dict[str, tuple[Element, int, int]]] = {}
        for page in pages:
            content = [
                e for e in parsed.elements if e.parent_id == page.id and e.kind in _CONTENT
            ]
            if not content:
                continue
            edges[page.id] = {}
            for position, element in (("header", content[0]), ("footer", content[-1])):
                line = self._edge_line(parsed, element, top=position == "header")
                if line:
                    edges[page.id][position] = (element, *line)

        def key(start: int, end: int) -> str:
            return re.sub(r"\d+", "#", " ".join(parsed.text[start:end].lower().split()))

        repeated = {
            position: {
                text
                for text, count in Counter(
                    key(found[position][1], found[position][2])
                    for found in edges.values()
                    if position in found
                ).items()
                if len(pages) >= self.min_pages and count >= self.repeat_share * len(pages)
            }
            for position in ("header", "footer")
        }
        replacements: dict[str, list[Element]] = {}
        for found in edges.values():
            for position, (element, start, end) in found.items():
                current = replacements.get(element.id, [element])
                target = current[-1] if position == "footer" else current[0]
                if not target.start <= start <= end <= target.end:
                    continue
                line = parsed.text[start:end]
                if not (_PAGE_NUMBER.match(line) or key(start, end) in repeated[position]):
                    continue
                marked = replace(
                    target,
                    id=f"{element.id}-{position}",
                    kind=position,
                    start=start,
                    end=end,
                    quality_flags=(*target.quality_flags, "page_furniture_heuristic"),
                )
                rest_start, rest_end = (
                    _trimmed(parsed.text, end, target.end)
                    if position == "header"
                    else _trimmed(parsed.text, target.start, start)
                )
                pieces = [marked]
                if rest_start < rest_end:
                    rest = replace(target, start=rest_start, end=rest_end)
                    pieces = [marked, rest] if position == "header" else [rest, marked]
                index = current.index(target)
                replacements[element.id] = current[:index] + pieces + current[index + 1 :]
        return _rebuild(parsed, replacements)

    @staticmethod
    def _edge_line(parsed: ParsedDocument, element: Element, *, top: bool) -> tuple[int, int] | None:
        body = parsed.text_of(element)
        lines = body.splitlines(keepends=True)
        offset = element.start
        spans = []
        for line in lines:
            if line.strip():
                spans.append(_trimmed(parsed.text, offset, offset + len(line)))
            offset += len(line)
        if not spans:
            return None
        return spans[0] if top else spans[-1]


class SplitNumberedClauses:
    """Give each numbered clause its own element.

    A PDF text layer has no clause boundaries, so several clauses arrive as one
    paragraph. Splitting at lines that begin "7." or "7.2" lets a chunker keep a
    clause whole and stop between clauses. A short numbered line that does not
    read as a sentence is promoted to a heading at the depth its number implies.
    """

    name = "split-numbered-clauses"

    def __init__(self, *, promote_titles: bool = True, max_title_length: int = 80) -> None:
        self.promote_titles = promote_titles
        self.max_title_length = max_title_length

    def clean(self, parsed: ParsedDocument) -> ParsedDocument:
        replacements: dict[str, list[Element]] = {}
        for element in parsed.elements:
            if element.kind != "paragraph":
                continue
            body = parsed.text_of(element)
            starts = [match.start() for match in _CLAUSE.finditer(body)]
            if not starts:
                continue
            cuts = sorted({0, *starts, len(body)})
            pieces = []
            for index, (begin, stop) in enumerate(zip(cuts, cuts[1:])):
                start, end = _trimmed(parsed.text, element.start + begin, element.start + stop)
                if start >= end:
                    continue
                piece = replace(element, id=f"{element.id}-clause-{index}", start=start, end=end)
                number = _CLAUSE.match(parsed.text[start:end])
                if number:
                    label = number.group(1).rstrip(".)")
                    text = parsed.text[start:end]
                    title = (
                        self.promote_titles
                        and "\n" not in text.strip()
                        and len(text) <= self.max_title_length
                        and not re.search(r"[.;,:]\s*$", text)
                    )
                    piece = replace(
                        piece,
                        kind="heading" if title else piece.kind,
                        level=label.count(".") + 1 if title else piece.level,
                        metadata={**piece.metadata, "number": label},
                        quality_flags=(*piece.quality_flags, "numbered_clause_heuristic"),
                    )
                pieces.append(piece)
            if len(pieces) > 1 or pieces and pieces[0].metadata.get("number"):
                replacements[element.id] = pieces
        return _rebuild(parsed, replacements)


class LinkAcrossPages:
    """Link a sentence that a page break cut in two.

    If the last content of a page does not end a sentence and the next page's
    content starts in lower case, the two elements are marked as one unit with
    ``continues`` and ``continued_from`` metadata. They stay separate spans, so
    any page furniture between them is not swallowed.
    """

    name = "link-across-pages"

    def clean(self, parsed: ParsedDocument) -> ParsedDocument:
        content = [e for e in parsed.elements if e.kind in _CONTENT and e.page_index is not None]
        links: dict[str, dict[str, str]] = {}
        for before, after in zip(content, content[1:]):
            if after.page_index != before.page_index + 1:
                continue
            first = parsed.text_of(after).lstrip()
            if _ENDS_SENTENCE.search(parsed.text_of(before)) or not first[:1].islower():
                continue
            links.setdefault(before.id, {})["continues"] = after.id
            links.setdefault(after.id, {})["continued_from"] = before.id
        if not links:
            return parsed
        return replace(
            parsed,
            elements=tuple(
                replace(e, metadata={**e.metadata, **links[e.id]}) if e.id in links else e
                for e in parsed.elements
            ),
        )

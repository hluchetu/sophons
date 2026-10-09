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
# A contents-page entry: dot leaders running to a page number.
_CONTENTS_ENTRY = re.compile(r"(?:\.\s?|…){3,}\s*\d+\s*$")
_CONTENTS_TITLE = re.compile(
    r"^\s*(?:table\s+of\s+contents|contents|arrangement\s+of\s+"
    r"(?:articles|sections|clauses|regulations|rules|paragraphs|orders))\s*$",
    re.IGNORECASE,
)
_ENDS_SENTENCE = re.compile(r"[.!?:;][\"'”’)\]]*\s*$")
_CONTENT = ("paragraph", "list_item")
# A heading's own number: "5", "5.0", "5.3.1", with or without a trailing dot.
_HEADING_NUMBER = re.compile(r"^\s*(\d{1,3}(?:\.\d{1,3})*)[.)]?\s+\S")


def _depth(number: str) -> int:
    """Outline depth of a clause number: 5 and 5.0 are 1, 5.3 is 2, 5.3.1 is 3."""

    parts = number.split(".")
    while len(parts) > 1 and parts[-1] == "0":
        parts.pop()
    return len(parts)


def _entry_key(line: str) -> str:
    """A heading as it reads in both a contents list and the body: no number, leaders or page."""

    text = re.sub(r"^\s*\d{1,3}[A-Za-z]?\s*[.)—–-]+\s*", "", line)
    text = re.sub(r"(?:[.…]\s?){2,}\s*\d*\s*$", "", text)
    return " ".join(text.lower().split()).strip(" .:;—–-")


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

    return [MarkPageFurniture(), MarkContents(), SplitNumberedClauses(), LinkAcrossPages()]


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


class MarkContents:
    """Reclassify a table of contents so it is not chunked or searched as content.

    Two patterns are recognised, both line by line:

    - a run of entries with dot leaders running to a page number;
    - a titled list ("Contents", "Arrangement of Sections") whose first entry
      appears again later as a heading: everything from the title up to that
      second appearance is the contents.

    A contents list repeats every heading in the document, so left in place it
    competes with the provisions themselves in search. The text is kept; the
    spans become ``contents`` elements.
    """

    name = "mark-contents"

    def __init__(self, *, min_entries: int = 3, max_share: float = 0.25) -> None:
        self.min_entries = min_entries
        self.max_share = max_share

    def clean(self, parsed: ParsedDocument) -> ParsedDocument:
        # Every nonblank line of content, in reading order: (element, start, end).
        lines: list[tuple[Element, int, int]] = []
        for element in parsed.elements:
            if element.kind not in _CONTENT:
                continue
            offset = element.start
            for line in parsed.text_of(element).splitlines(keepends=True):
                if line.strip():
                    lines.append((element, *_trimmed(parsed.text, offset, offset + len(line))))
                offset += len(line)
        if not lines:
            return parsed
        texts = [parsed.text[start:end] for _, start, end in lines]
        marked = [False] * len(lines)
        for first, last in (*self._leader_runs(texts), *self._titled_lists(texts, lines, parsed)):
            for index in range(first, last):
                marked[index] = True
        if not any(marked):
            return parsed

        replacements: dict[str, list[Element]] = {}
        index = 0
        while index < len(lines):
            element = lines[index][0]
            stop = index
            while stop < len(lines) and lines[stop][0] is element:
                stop += 1
            if any(marked[index:stop]):
                replacements[element.id] = self._split(element, lines[index:stop], marked[index:stop])
            index = stop
        return _rebuild(parsed, replacements)

    def _leader_runs(self, texts: list[str]) -> list[tuple[int, int]]:
        """Runs of dot-leader entries; a wrapped title or a bare part heading may sit inside."""

        runs, index = [], 0
        entries = [bool(_CONTENTS_ENTRY.search(text)) for text in texts]
        while index < len(texts):
            if not entries[index]:
                index += 1
                continue
            last, count, cursor = index, 1, index + 1
            while cursor < len(texts) and cursor - last <= 3:
                if entries[cursor]:
                    last, count = cursor, count + 1
                cursor += 1
            if count >= self.min_entries:
                first = index
                while first > 0 and index - first < 2 and _CONTENTS_TITLE.match(texts[first - 1]):
                    first -= 1
                if first > 0 and _CONTENTS_TITLE.match(texts[first - 1]):
                    first -= 1
                runs.append((first, last + 1))
            index = last + 1
        return runs

    def _titled_lists(
        self, texts: list[str], lines: list[tuple[Element, int, int]], parsed: ParsedDocument
    ) -> list[tuple[int, int]]:
        found = []
        keys = [_entry_key(text) for text in texts]
        for title, text in enumerate(texts):
            if not _CONTENTS_TITLE.match(text):
                continue
            for entry in range(title + 1, min(title + 4, len(texts))):
                if len(keys[entry]) < 4:
                    continue
                again = next(
                    (
                        later
                        for later in range(title + 1 + self.min_entries, len(texts))
                        if keys[later] == keys[entry]
                    ),
                    None,
                )
                if again is None:
                    continue
                length = lines[again][1] - lines[title][1]
                listed = texts[title + 1 : again]
                # A list of headings: short lines, and a modest part of the document.
                if length <= self.max_share * len(parsed.text) and all(
                    len(item) <= 200 for item in listed
                ):
                    found.append((title, again))
                break
        return found

    @staticmethod
    def _split(
        element: Element, lines: list[tuple[Element, int, int]], marked: list[bool]
    ) -> list[Element]:
        pieces: list[Element] = []
        index = 0
        while index < len(lines):
            stop = index
            while stop < len(lines) and marked[stop] == marked[index]:
                stop += 1
            start, end = lines[index][1], lines[stop - 1][2]
            if marked[index]:
                pieces.append(
                    replace(
                        element,
                        id=f"{element.id}-contents-{len(pieces)}",
                        kind="contents",
                        start=start,
                        end=end,
                        quality_flags=(*element.quality_flags, "contents_heuristic"),
                    )
                )
            else:
                kept = any(piece.id == element.id for piece in pieces)
                pieces.append(
                    replace(
                        element,
                        id=f"{element.id}-part-{len(pieces)}" if kept else element.id,
                        start=start,
                        end=end,
                    )
                )
            index = stop
        return pieces


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
        outline = 0  # depth of the numbered section currently open, 0 before any
        for element in parsed.elements:
            if element.kind == "heading":
                # A heading another parser found: take its depth from its number.
                number = _HEADING_NUMBER.match(parsed.text_of(element))
                if number:
                    label = number.group(1)
                    outline = _depth(label)
                    replacements[element.id] = [
                        replace(
                            element,
                            level=outline,
                            metadata={**element.metadata, "number": label},
                        )
                    ]
                elif outline:
                    # An unnumbered heading inside a numbered outline sits under the
                    # open section; left at its own level it would close that section.
                    replacements[element.id] = [
                        replace(
                            element,
                            level=outline + 1,
                            quality_flags=(*element.quality_flags, "heading_level_heuristic"),
                        )
                    ]
                continue
            if element.kind not in _CONTENT:
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
                        and not _CONTENTS_ENTRY.search(text)
                    )
                    piece = replace(
                        piece,
                        kind="heading" if title else piece.kind,
                        level=_depth(label) if title else piece.level,
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

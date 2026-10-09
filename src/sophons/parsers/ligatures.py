"""Put back the letters a PDF font replaced with a ligature glyph.

Some PDFs draw "fi", "ff" and the like with one glyph and give it no Unicode
meaning, so text extraction returns a private-use character: "filed" comes out as
"\\ue000led". The word can then not be searched for, and no quotation of it matches.
Which character stands for which letters differs from font to font, so the
mapping is worked out per document from the words it would produce.
"""

from __future__ import annotations

import re
from collections import Counter

# Ligatures that do have a Unicode code point.
_STANDARD = {
    "ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl",
    "ﬅ": "st", "ﬆ": "st",
}  # fmt: skip
_CANDIDATES = ("fi", "fl", "ff", "ffi", "ffl")
_PRIVATE = re.compile(r"[-]")
_WORD = re.compile(r"[A-Za-z-]+")
# Common English and legal words written with these ligatures. A repaired word that is
# one of these, or one of these with a plain ending, counts as evidence for a mapping.
_KNOWN = frozenset(
    """
    file filed files filing final finally finalise finalize find finding findings finds fine
    fined fines finish finished firm firms first fiscal fit five fix fixed field fields figure
    figures fifth fifty fight fill filled finance financial fire fiduciary benefit benefits
    beneficiary beneficial certificate certificates certified certify certification confirm
    confirmed confirmation confine confined define defined definition definite definitely
    deficiency deficit specific specifically specified specify specification justify justified
    justification satisfy satisfied satisfaction dissatisfied testify testified notify notified
    notification identify identified identification classify classified clarify clarified
    verify verified verification modify modified modification qualify qualified qualification
    ratify ratified rectify rectified profit profits significant significantly scientific
    artificial sacrifice fortified unfit infinite fiction fidelity
    office offices officer officers official officially sufficient sufficiently insufficient
    efficient efficiency difficult difficulty difficulties affidavit affidavits affirm affirmed
    affirmation traffic trafficking affiliate affiliated affix affixed
    staff effort efforts effect effects effective effectively affect affected affects offer
    offered offers offence offences offend offender different difference differ differed afford
    afforded suffer suffered suffering plaintiff plaintiffs sheriff tariff bailiff off offset
    buffer stuff affair affairs
    conflict conflicts flaw flawed flow flows floor flat flexible flight float flood reflect
    reflected reflects influence influenced inflict inflicted inflation flag flagrant flee fled
    fly flying fluid fluctuate overflow
    affluent baffled shuffle
    """.split()
)
_ENDINGS = ("", "s", "es", "ed", "d", "ing", "ly")


def _known(word: str) -> bool:
    word = word.lower()
    return any(
        (word[: -len(ending)] if ending else word) in _KNOWN
        for ending in _ENDINGS
        if word.endswith(ending)
    )


def repair_ligatures(pages: list[str]) -> tuple[list[str], bool]:
    """The pages with ligature glyphs spelled out, and whether anything changed.

    A private-use character is replaced only when one spelling turns the words it
    appears in into known words more often than any other; otherwise it is left.
    """

    mapping = dict(_STANDARD)
    words: dict[str, Counter[str]] = {}
    for page in pages:
        if not _PRIVATE.search(page):
            continue
        for word in _WORD.findall(page):
            for glyph in set(_PRIVATE.findall(word)):
                words.setdefault(glyph, Counter())[word] += 1
    # Settle the glyphs with the clearest evidence first: a word holding two different
    # glyphs can only be judged once one of them is known.
    for _ in range(len(words)):
        best: tuple[int, str, str] | None = None
        for glyph, found in words.items():
            if glyph in mapping:
                continue
            scores = []
            for letters in _CANDIDATES:
                trial = {**mapping, glyph: letters}
                scores.append(
                    (
                        sum(
                            count
                            for word, count in found.items()
                            if _known("".join(trial.get(char, char) for char in word))
                        ),
                        letters,
                    )
                )
            scores.sort(reverse=True)
            if scores[0][0] > scores[1][0] and (best is None or scores[0][0] > best[0]):
                best = (scores[0][0], glyph, scores[0][1])
        if best is None:
            break
        mapping[best[1]] = best[2]
    if not any(char in page for page in pages for char in mapping):
        return pages, False
    table = str.maketrans(mapping)
    return [page.translate(table) for page in pages], True

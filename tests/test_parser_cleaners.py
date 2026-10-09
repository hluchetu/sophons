import pytest

from sophons.parsers import (
    Blob,
    Cleaner,
    LinkAcrossPages,
    MarkdownParser,
    MarkPageFurniture,
    SplitNumberedClauses,
    clean,
    default_cleaners,
    page_document,
)
from sophons.splitters import StructureAwareSplitter

PAGE_ONE = (
    "- 35 -\nAPPENDIX 1\nTENANCY AGREEMENT\nTHIS TENANCY AGREEMENT is made this day.\n"
    "4. The Tenant shall pay the rent.\n"
    "5. The Tenant shall at all time keep the interior of\n"
    "the premises including the doors, windows, sanitary and apparatus,"
)
PAGE_TWO = (
    "- 36 -\nfittings and the electrical wiring apparatus in order and shall hand over\n"
    "the premises in the same manner on vacation.\n"
    "6. That the tenant shall not subject, part possession with the whole or\n"
    "part of the leased premises.\n"
    "10. That the tenant shall not sublet or transfer to any other person but\n"
    "will surrender the premises to the landlord."
)


def pdf(*pages):
    blob = Blob(b"x", "application/pdf", source="lease.pdf")
    return page_document(blob, [(page, "pdf") for page in pages], parser="pypdf")


def kinds(parsed):
    return [(e.kind, parsed.text_of(e).splitlines()[0]) for e in parsed.elements if e.kind != "page"]


def test_cleaning_never_changes_the_text_and_always_validates():
    parsed = pdf(PAGE_ONE, PAGE_TWO)
    cleaned = clean(parsed)
    assert cleaned.text == parsed.text
    cleaned.validate()
    assert all(isinstance(cleaner, Cleaner) for cleaner in default_cleaners())
    # Offsets recorded before cleaning still point at the same words.
    start = parsed.text.index("shall not sublet")
    assert cleaned.text[start : start + 16] == "shall not sublet"


def test_page_numbers_become_furniture_and_leave_the_content():
    cleaned = MarkPageFurniture().clean(pdf(PAGE_ONE, PAGE_TWO)).validate()
    furniture = [e for e in cleaned.elements if e.kind == "header"]
    assert [cleaned.text_of(e) for e in furniture] == ["- 35 -", "- 36 -"]
    assert all("page_furniture_heuristic" in e.quality_flags for e in furniture)
    assert all("- 3" not in cleaned.text_of(e) for e in cleaned.leaves())
    assert len(cleaned.leaves(include_furniture=True)) == len(cleaned.leaves()) + 2


def test_running_headers_and_footers_are_found_by_repetition_not_by_guessing():
    pages = [f"Republic of Kenya Gazette\nBody text of page {n} goes here.\nPage {n} of 4" for n in (1, 2, 3, 4)]
    cleaned = MarkPageFurniture().clean(pdf(*pages)).validate()
    assert [cleaned.text_of(e) for e in cleaned.elements if e.kind == "header"] == ["Republic of Kenya Gazette"] * 4
    assert [cleaned.text_of(e) for e in cleaned.elements if e.kind == "footer"] == [
        f"Page {n} of 4" for n in (1, 2, 3, 4)
    ]
    assert [cleaned.text_of(e) for e in cleaned.leaves()] == [
        f"Body text of page {n} goes here." for n in (1, 2, 3, 4)
    ]
    # Two pages are too few to call a repeated first line a running header.
    two = MarkPageFurniture().clean(pdf("Same title\nFirst body.", "Same title\nSecond body."))
    assert not [e for e in two.elements if e.kind in ("header", "footer")]
    # A document without pages is returned untouched.
    markdown = MarkdownParser().parse(Blob(b"# T\n\n12\n", "text/markdown"))
    assert MarkPageFurniture().clean(markdown) is markdown


def test_numbered_clauses_get_their_own_elements():
    cleaned = SplitNumberedClauses().clean(MarkPageFurniture().clean(pdf(PAGE_ONE, PAGE_TWO))).validate()
    numbered = [(e.metadata["number"], cleaned.text_of(e).splitlines()[0][:22]) for e in cleaned.elements if "number" in e.metadata]
    assert numbered == [
        ("4", "4. The Tenant shall pa"),
        ("5", "5. The Tenant shall at"),
        ("6", "6. That the tenant sha"),
        ("10", "10. That the tenant sh"),
    ]
    assert all(e.kind == "paragraph" for e in cleaned.elements if "number" in e.metadata)  # sentences, not titles
    preamble = next(e for e in cleaned.elements if cleaned.text_of(e).startswith("APPENDIX"))
    assert "number" not in preamble.metadata and "4." not in cleaned.text_of(preamble)


def test_short_numbered_titles_become_headings_and_amounts_are_left_alone():
    text = (
        "7. Termination\n7.1 Either party may end the tenancy on notice.\n7.2 Notice\n"
        "232 – 01020 Kenol is the address.\nIn 2025 the rent was 45,000.\n8) Final clause."
    )
    cleaned = SplitNumberedClauses().clean(pdf(text)).validate()
    found = [(e.kind, e.level, e.metadata.get("number")) for e in cleaned.elements if e.kind != "page"]
    assert found == [
        ("heading", 1, "7"),
        ("paragraph", None, "7.1"),
        # The address and year lines are not clause starts; they stay in 7.2's text,
        # which therefore reads as a clause with a body, not a bare title.
        ("paragraph", None, "7.2"),
        ("paragraph", None, "8"),
    ]
    nested = SplitNumberedClauses().clean(pdf("7. Termination\n7.1 Notice\n7.2 Either party may end it."))
    assert [(e.kind, e.level) for e in nested.elements if e.kind == "heading"] == [("heading", 1), ("heading", 2)]
    contents = SplitNumberedClauses().clean(
        pdf("1.0 INTRODUCTION……………......…………...10\n2.0 RECRUITMENT . . . . . . 12\n3.0 LEAVE")
    )
    # Contents-page entries are numbered lines, but they are not section headings.
    assert [contents.text_of(e) for e in contents.elements if e.kind == "heading"] == ["3.0 LEAVE"]
    plain = SplitNumberedClauses(promote_titles=False).clean(pdf(text))
    assert not [e for e in plain.elements if e.kind == "heading"]
    untouched = pdf("No numbering here.\n\nJust two paragraphs.")
    assert SplitNumberedClauses().clean(untouched) == untouched


def test_a_sentence_cut_by_a_page_break_is_linked_not_merged():
    cleaned = clean(pdf(PAGE_ONE, PAGE_TWO))
    head = next(e for e in cleaned.elements if e.metadata.get("number") == "5")
    tail = next(e for e in cleaned.elements if e.metadata.get("continued_from") == head.id)
    assert head.metadata["continues"] == tail.id
    assert cleaned.text_of(tail).startswith("fittings and the electrical")
    assert (head.page_index, tail.page_index) == (0, 1)
    assert "- 36 -" not in cleaned.text_of(head) + cleaned.text_of(tail)
    # A sentence that ends, or a next page starting a new clause, is not linked.
    separate = LinkAcrossPages().clean(pdf("The clause ends here.", "next words in lower case"))
    assert not any(e.metadata for e in separate.elements)
    fresh = LinkAcrossPages().clean(pdf("An unfinished line,", "6. A new clause begins."))
    assert not any(e.metadata for e in fresh.elements)


def test_chunks_from_cleaned_text_keep_clauses_whole_and_rejoin_split_sentences():
    cleaned = clean(pdf(PAGE_ONE, PAGE_TWO))
    chunks = StructureAwareSplitter(max_chunk_size=260).split_parsed(cleaned)
    assert all("- 35 -" not in c.content and "- 36 -" not in c.content for c in chunks)
    joined = next(c for c in chunks if "5. The Tenant" in c.content)
    assert "sanitary and apparatus, fittings and the electrical wiring" in joined.content
    assert joined.metadata["pages"] == [1, 2]
    last = chunks[-1]
    assert last.content.startswith("6. That the tenant") and "10. That the tenant" in last.content
    assert last.metadata["pages"] == [2]
    assert "numbered_clause_heuristic" in last.metadata["quality_flags"]
    # A tiny size target still never separates the two halves of the split sentence.
    tight = StructureAwareSplitter(max_chunk_size=40).split_parsed(cleaned)
    assert any("apparatus, fittings" in c.content for c in tight)


def test_cleaners_run_in_the_order_given_and_each_result_is_checked():
    class Breaks:
        name = "breaks"

        def clean(self, parsed):
            from dataclasses import replace

            return replace(parsed, elements=(parsed.elements[1], parsed.elements[0]))

    from sophons.parsers import InvalidParsedDocument

    with pytest.raises(InvalidParsedDocument):
        clean(pdf(PAGE_ONE), [Breaks()])
    assert clean(pdf(PAGE_ONE), []) == pdf(PAGE_ONE)

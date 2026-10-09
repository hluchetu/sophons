import sys
import types

import pytest

from sophons.errors import LoaderError, MissingDependencyError
from sophons.parsers import Blob, DoclingParser, SplitNumberedClauses, clean, default_registry
from sophons.splitters import StructureAwareSplitter


def item(label, text="", *, page=1, level=None, cells=None, box=(1.0, 2.0, 3.0, 4.0)):
    bbox = types.SimpleNamespace(l=box[0], t=box[1], r=box[2], b=box[3])
    fields = {"label": types.SimpleNamespace(value=label), "text": text,
              "prov": [types.SimpleNamespace(page_no=page, bbox=bbox)] if page else []}
    if level is not None:
        fields["level"] = level
    if cells is not None:
        fields["data"] = types.SimpleNamespace(table_cells=cells)
    return types.SimpleNamespace(**fields)


def cell(text, row, column, *, row_span=1, col_span=1, header=False):
    return types.SimpleNamespace(text=text, start_row_offset_idx=row, start_col_offset_idx=column,
                                 row_span=row_span, col_span=col_span, column_header=header)


ITEMS = [
    item("page_header", "Republic of Kenya"),
    item("title", "Tenancy Agreement"),
    item("document_index", cells=[cell("1.0 Rent", 0, 0), cell("3", 0, 1)]),
    item("section_header", "1.0 RENT", level=1),
    item("text", "The rent is payable monthly and"),
    item("page_footer", "- 35 -"),
    item("text", "in advance on the fifth day.", page=2),
    item("section_header", "1.1 Deposit", page=2, level=1),
    item("list_item", "a) two months' rent", page=2),
    item("picture", "", page=2),
    item("section_header", "Refund", page=2, level=1),
    item("table", page=2, cells=[cell("Item", 0, 0, header=True), cell("Cost", 0, 1, header=True),
                                 cell("Deposit  held", 1, 0, col_span=2)]),
    item("formula", "x = y", page=None),
]


@pytest.fixture
def docling(monkeypatch):
    """A stand-in converter and the one Docling type the parser constructs."""
    seen = {}

    class Stream:
        def __init__(self, name, stream):
            seen["name"], seen["bytes"] = name, stream.read()

    class Converter:
        def convert(self, source, max_num_pages):
            seen["max_pages"] = max_num_pages
            if seen.get("fail"):
                raise RuntimeError("backend exploded")
            document = types.SimpleNamespace(iterate_items=lambda **kw: ((i, 0) for i in ITEMS))
            return types.SimpleNamespace(document=document)

    base = types.SimpleNamespace(DocumentStream=Stream)
    monkeypatch.setitem(sys.modules, "docling", types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, "docling.datamodel", types.SimpleNamespace(base_models=base))
    monkeypatch.setitem(sys.modules, "docling.datamodel.base_models", base)
    monkeypatch.setitem(sys.modules, "docling_core.types.doc", None)
    seen["converter"] = Converter()
    return seen


def test_docling_items_become_typed_elements_that_honour_the_contract(docling):
    blob = Blob(b"%PDF", "application/pdf", source="/tmp/lease.pdf", metadata={"matter": "m"})
    parsed = DoclingParser(converter=docling["converter"]).parse(blob).validate()
    assert (docling["name"], docling["bytes"], docling["max_pages"]) == ("lease.pdf", b"%PDF", 500)
    assert parsed.parser == "docling" and parsed.metadata == {"matter": "m"}
    found = [(e.kind, parsed.text_of(e)) for e in parsed.elements if e.kind not in ("row", "cell")]
    assert found == [
        ("header", "Republic of Kenya"),
        ("heading", "Tenancy Agreement"),
        ("contents", "1.0 Rent\t3"),
        ("heading", "1.0 RENT"),
        ("paragraph", "The rent is payable monthly and"),
        ("footer", "- 35 -"),
        ("paragraph", "in advance on the fifth day."),
        ("heading", "1.1 Deposit"),
        ("list_item", "a) two months' rent"),
        ("heading", "Refund"),
        ("table", "Item\tCost\nDeposit held"),
        ("paragraph", "x = y"),  # an unmapped label keeps its text; the empty picture is dropped
    ]
    assert all(e.method == "layout" for e in parsed.elements)
    first = parsed.elements[0]
    assert first.page_index == 0 and first.metadata == {"docling_label": "page_header", "bbox": [1.0, 2.0, 3.0, 4.0]}
    assert parsed.elements[-1].page_index is None
    table = next(e for e in parsed.elements if e.kind == "table")
    assert table.quality_flags == ("merged_cells_unresolved",)
    cells = [(e.row_index, e.column_index, parsed.text_of(e)) for e in parsed.elements
             if e.kind == "cell" and e.id.startswith(table.id)]
    assert cells == [(0, 0, "Item"), (0, 1, "Cost"), (1, 0, "Deposit held")]
    assert parsed.elements[[e.id for e in parsed.elements].index(f"{table.id}-row-0-cell-0")].metadata == {"header": True}
    # Furniture and the contents table are addressable but are not content.
    assert [e.kind for e in parsed.leaves()].count("contents") == 0
    assert "Republic of Kenya" not in parsed.to_markdown() and "| Item | Cost |" in parsed.to_markdown()


def test_cleaners_add_outline_depth_and_page_links_to_docling_output(docling):
    parsed = DoclingParser(converter=docling["converter"]).parse(Blob(b"%PDF", "application/pdf"))
    cleaned = clean(parsed)
    assert cleaned.text == parsed.text
    levels = {cleaned.text_of(e): (e.level, e.metadata.get("number")) for e in cleaned.elements if e.kind == "heading"}
    assert levels == {
        "Tenancy Agreement": (1, None),  # before the numbering starts: untouched
        "1.0 RENT": (1, "1.0"),
        "1.1 Deposit": (2, "1.1"),
        "Refund": (3, None),  # unnumbered, so it sits under the open section
    }
    refund = next(e for e in cleaned.elements if cleaned.text_of(e) == "Refund")
    assert "heading_level_heuristic" in refund.quality_flags
    table = next(e for e in cleaned.elements if e.kind == "table")
    assert cleaned.heading_paths()[table.id] == ("1.0 RENT", "1.1 Deposit", "Refund")
    head = next(e for e in cleaned.elements if cleaned.text_of(e).endswith("monthly and"))
    assert cleaned.text_of(next(e for e in cleaned.elements if e.id == head.metadata["continues"])).startswith("in advance")
    chunks = StructureAwareSplitter().split_parsed(cleaned)
    rent = next(c for c in chunks if "payable monthly" in c.content)
    assert "monthly and in advance on the fifth day." in rent.content and "- 35 -" not in rent.content
    assert rent.metadata["pages"] == [1, 2] and rent.metadata["methods"] == ["layout"]
    assert SplitNumberedClauses().clean(parsed).text == parsed.text


def test_stream_name_gets_a_suffix_and_failures_are_reported(docling):
    parser = DoclingParser(converter=docling["converter"], max_pages=7)
    parser.parse(Blob(b"%PDF", "application/pdf"))
    assert (docling["name"], docling["max_pages"]) == ("document.pdf", 7)
    parser.parse(Blob(b"PK", "application/vnd.openxmlformats-officedocument.wordprocessingml.document", source="upload"))
    assert docling["name"] == "upload.docx"
    docling["fail"] = True
    with pytest.raises(LoaderError, match="could not convert"):
        parser.parse(Blob(b"%PDF", "application/pdf"))
    with pytest.raises(LoaderError, match="empty"):
        parser.parse(Blob(b"", "application/pdf"))


def test_docling_is_opt_in_and_its_absence_is_named(monkeypatch):
    assert [p.name for p in default_registry().parsers_for("application/pdf")] == ["pypdf", "ocr"]
    assert [p.name for p in default_registry(layout=True).parsers_for("application/pdf")] == ["docling", "pypdf", "ocr"]
    for name in ("docling", "docling.datamodel", "docling.datamodel.base_models", "docling.document_converter"):
        monkeypatch.setitem(sys.modules, name, None)
    with pytest.raises(MissingDependencyError) as missing:
        default_registry(layout=True).parse(Blob(b"%PDF", "application/pdf"))
    assert missing.value.details == {"dependency": "docling", "extra": "docling"}

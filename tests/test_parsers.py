import io
import sys
import types
import zipfile

import pytest

from sophons.errors import LoaderError, MissingDependencyError, UnsupportedFileTypeError
from sophons.loaders import DocxLoader, FileLoader
from sophons.parsers import (
    Blob,
    DocxParser,
    Element,
    InvalidParsedDocument,
    MarkdownParser,
    ParsedDocument,
    Parser,
    ParserDeclined,
    ParserRegistry,
    PlainTextParser,
    PyPDFParser,
    default_registry,
    guess_mime_type,
)
from sophons.splitters import StructureAwareSplitter

DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"

MARKDOWN = (
    "# Lease\n\nThe tenant pays rent\nmonthly.\n\n## Terms\n- no subletting\n  at all\n- no pets\n\n"
    "| Item | Cost |\n| --- | --- |\n| Rent | 45,000 |\n\n```py\n# not a heading\n```\n"
)


def docx_bytes(merged: bool = False) -> bytes:
    def paragraph(text, style=None, numbered=False):
        properties = (f'<w:pStyle w:val="{style}"/>' if style else "") + ("<w:numPr/>" if numbered else "")
        return f"<w:p><w:pPr>{properties}</w:pPr><w:r><w:t>{text}</w:t></w:r></w:p>"

    def cell(text, span=False):
        return f"<w:tc>{'<w:tcPr><w:gridSpan w:val=\"2\"/></w:tcPr>' if span else ''}{paragraph(text)}</w:tc>"

    body = (
        paragraph("Tenancy Agreement", "Ttl")
        + paragraph("1. Rent", "H2")
        + paragraph("The rent is KES 45,000.")
        + paragraph("")
        + paragraph("Paid monthly", numbered=True)
        + f"<w:tbl><w:tr>{cell('Item')}{cell('Cost')}</w:tr><w:tr>{cell('Rent', merged)}{cell('45,000')}</w:tr></w:tbl>"
        + f"<w:sdt><w:sdtContent>{paragraph('Signed by the parties.')}</w:sdtContent></w:sdt>"
    )
    styles = (
        f'<w:styles xmlns:w="{W}"><w:style w:styleId="Ttl"><w:name w:val="Title"/></w:style>'
        '<w:style w:styleId="H2"><w:name w:val="heading 2"/></w:style></w:styles>'
    )
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr("word/document.xml", f'<w:document xmlns:w="{W}"><w:body>{body}</w:body></w:document>')
        archive.writestr("word/styles.xml", styles)
    return output.getvalue()


@pytest.fixture
def fake_pypdf(monkeypatch):
    """A stand-in pypdf whose pages come from the test."""
    state = {"pages": ["First page.\n\nSecond paragraph.", "Clause 10 continues."], "encrypted": False,
             "labels": None}

    class Reader:
        def __init__(self, stream):
            self.is_encrypted = state["encrypted"]
            self.pages = [types.SimpleNamespace(extract_text=lambda text=text: text) for text in state["pages"]]
            self.trailer = {"/Root": {"/PageLabels": True} if state["labels"] else {}}
            self.page_labels = state["labels"]

    monkeypatch.setitem(sys.modules, "pypdf", types.SimpleNamespace(PdfReader=Reader))
    return state


def parsed_samples(fake_pypdf=None):
    return [
        PlainTextParser().parse(Blob(b"One.\n\nTwo\nlines.\n\n\nThree.", "text/plain", source="a.txt")),
        MarkdownParser().parse(Blob(MARKDOWN.encode(), "text/markdown", source="a.md")),
        DocxParser().parse(Blob(docx_bytes(), DOCX, source="a.docx")),
    ]


@pytest.mark.parametrize("parsed", parsed_samples(), ids=["text", "markdown", "docx"])
def test_every_parser_honours_the_shared_contract(parsed):
    assert isinstance(parsed, ParsedDocument)
    parsed.validate()
    assert parsed.parser and parsed.mime_type and len(parsed.checksum) == 64
    ids = [element.id for element in parsed.elements]
    assert len(ids) == len(set(ids)) and parsed.elements
    for element in parsed.elements:
        assert 0 <= element.start <= element.end <= len(parsed.text)
    # Content units are non-empty, and nothing is lost crossing a process boundary.
    assert all(parsed.text_of(element).strip() for element in parsed.leaves())
    assert ParsedDocument.from_dict(parsed.to_dict()) == parsed
    flat = parsed.to_document()
    assert flat.content == parsed.text and flat.metadata["parser"] == parsed.parser


def test_pypdf_parser_honours_the_contract_and_keeps_pages(fake_pypdf):
    fake_pypdf["labels"] = ["35", "36"]
    parsed = PyPDFParser().parse(Blob(b"%PDF", "application/pdf", source="lease.pdf"))
    parsed.validate()
    pages = [e for e in parsed.elements if e.kind == "page"]
    assert [(p.page_index, p.page_label) for p in pages] == [(0, "35"), (1, "36")]
    paragraphs = [e for e in parsed.elements if e.kind == "paragraph"]
    assert [parsed.text_of(p) for p in paragraphs] == ["First page.", "Second paragraph.", "Clause 10 continues."]
    assert paragraphs[2].parent_id == "page-1" and paragraphs[2].page_index == 1
    assert "paragraph_boundary_heuristic" in paragraphs[0].quality_flags
    assert parsed.page_indexes(paragraphs[1].start, paragraphs[2].end) == [0, 1]
    assert ParsedDocument.from_dict(parsed.to_dict()) == parsed


def test_pypdf_parser_declines_scans_and_refuses_bad_input(fake_pypdf, monkeypatch):
    fake_pypdf["pages"] = ["Text", "   "]
    with pytest.raises(ParserDeclined, match="without a text layer"):
        PyPDFParser().parse(Blob(b"%PDF", "application/pdf"))
    kept = PyPDFParser(allow_empty_pages=True).parse(Blob(b"%PDF", "application/pdf"))
    assert "empty_page" in kept.elements[-1].quality_flags
    fake_pypdf["encrypted"] = True
    with pytest.raises(LoaderError, match="Password-protected"):
        PyPDFParser().parse(Blob(b"%PDF", "application/pdf"))
    fake_pypdf.update(encrypted=False, pages=["x"] * 3)
    with pytest.raises(LoaderError, match="no more than 2 pages"):
        PyPDFParser(max_pages=2).parse(Blob(b"%PDF", "application/pdf"))
    monkeypatch.setitem(sys.modules, "pypdf", None)
    with pytest.raises(MissingDependencyError):
        PyPDFParser().parse(Blob(b"%PDF", "application/pdf"))


def test_markdown_structure_levels_lists_tables_and_code():
    parsed = MarkdownParser().parse(Blob(MARKDOWN.encode(), "text/markdown"))
    kinds = [(e.kind, parsed.text_of(e)) for e in parsed.leaves()]
    assert kinds[0] == ("heading", "Lease") and kinds[2] == ("heading", "Terms")
    assert kinds[1] == ("paragraph", "The tenant pays rent\nmonthly.")
    assert kinds[3] == ("list_item", "- no subletting\n  at all") and kinds[4][1] == "- no pets"
    assert [e.level for e in parsed.elements if e.kind == "heading"] == [1, 2]
    cells = [parsed.text_of(e) for e in parsed.elements if e.kind == "cell"]
    assert cells == ["Item", "Cost", "Rent", "45,000"]  # the rule row is not data
    code = next(e for e in parsed.elements if e.kind == "code")
    assert parsed.text_of(code) == "```py\n# not a heading\n```"
    assert parsed.heading_paths()[code.id] == ("Lease", "Terms")
    assert parsed.to_markdown().startswith("# Lease\n\nThe tenant pays rent")
    assert "| Rent | 45,000 |" in parsed.to_markdown()


def test_docx_headings_come_from_styles_and_tables_keep_cells():
    parsed = DocxParser().parse(Blob(docx_bytes(merged=True), DOCX))
    leaves = [(e.kind, e.level, parsed.text_of(e)) for e in parsed.leaves()]
    assert leaves[:4] == [
        ("heading", 1, "Tenancy Agreement"),
        ("heading", 2, "1. Rent"),
        ("paragraph", None, "The rent is KES 45,000."),
        ("list_item", None, "Paid monthly"),  # the empty paragraph before it is skipped
    ]
    table = next(e for e in parsed.elements if e.kind == "table")
    assert parsed.text_of(table) == "Item\tCost\nRent\t45,000"
    assert table.quality_flags == ("merged_cells_unresolved",)
    cell = next(e for e in parsed.elements if e.id.endswith("row-1-cell-1"))
    assert (parsed.text_of(cell), cell.row_index, cell.column_index) == ("45,000", 1, 1)
    assert leaves[-1] == ("paragraph", None, "Signed by the parties.")  # inside a content control
    assert parsed.heading_paths()[table.id] == ("Tenancy Agreement", "1. Rent")
    assert DocxParser().parse(Blob(docx_bytes(), DOCX)).elements[4].quality_flags == ()
    with pytest.raises(LoaderError, match="not a readable DOCX"):
        DocxParser().parse(Blob(b"not a zip", DOCX))


def test_validation_rejects_inconsistent_elements():
    def document(*elements):
        return ParsedDocument(text="abcdef", elements=elements, mime_type="text/plain", parser="t")

    good = Element(id="a", kind="paragraph", start=0, end=3)
    document(good).validate()
    for bad in (
        (good, Element(id="a", kind="paragraph", start=3, end=4)),  # duplicate ID
        (Element(id="x", kind="paragraph", start=2, end=9),),  # beyond the text
        (Element(id="x", kind="nonsense", start=0, end=1),),
        (Element(id="x", kind="paragraph", start=3, end=4), Element(id="y", kind="paragraph", start=0, end=1)),
        (Element(id="c", kind="cell", start=0, end=1, parent_id="missing"),),
        (Element(id="p", kind="page", start=0, end=2), Element(id="c", kind="paragraph", start=1, end=4, parent_id="p")),
        (Element(id="h", kind="heading", start=0, end=1, level=0),),
    ):
        with pytest.raises(InvalidParsedDocument):
            document(*bad).validate()


def test_registry_routes_by_type_falls_through_and_reports():
    class Declines:
        name = "declines"

        def parse(self, blob):
            raise ParserDeclined("cannot read this one")

    registry = default_registry()
    assert isinstance(PlainTextParser(), Parser)
    assert registry.parse(Blob(b"Hello", "text/plain; charset=utf-8")).parser == "plain-text"
    registry.register("text/plain", Declines(), first=True)
    assert registry.parse(Blob(b"Hello", "text/plain")).parser == "plain-text"  # next in line
    only = ParserRegistry()
    only.register("text/plain", Declines())
    with pytest.raises(ParserDeclined) as declined:
        only.parse(Blob(b"Hello", "text/plain"))
    assert declined.value.details == {"declined": ["declines: cannot read this one"]}
    with pytest.raises(UnsupportedFileTypeError):
        registry.parse(Blob(b"x", "application/zip"))
    family = ParserRegistry()
    family.register("text/*", PlainTextParser())
    assert family.parse(Blob(b"a,b", "text/csv")).parser == "plain-text"


def test_blob_detects_types_and_refuses_unknown(tmp_path):
    assert guess_mime_type("Lease.DOCX") == DOCX and guess_mime_type("notes.md") == "text/markdown"
    path = tmp_path / "note.md"
    path.write_text("# Hi")
    blob = Blob.from_path(path)
    assert (blob.mime_type, blob.source, blob.data) == ("text/markdown", str(path), b"# Hi")
    unknown = tmp_path / "data.qqq"
    unknown.write_bytes(b"x")
    with pytest.raises(ValueError, match="mime type"):
        Blob.from_path(unknown)
    with pytest.raises(LoaderError, match="UTF-8"):
        PlainTextParser().parse(Blob(b"\xff\xfe\x00", "text/plain"))


def test_splitter_packs_elements_with_offsets_pages_and_flags(fake_pypdf):
    parsed = MarkdownParser().parse(Blob(MARKDOWN.encode(), "text/markdown", source="lease.md"))
    chunks = StructureAwareSplitter(max_chunk_size=1000).split_parsed(parsed)
    assert [chunk.metadata["heading_path"] for chunk in chunks] == [["Lease"], ["Lease", "Terms"]]
    terms = chunks[1]
    assert terms.id == "lease.md#structure_1" and terms.metadata["parser"] == "markdown"
    assert terms.metadata["block_types"] == ["heading", "list_item", "list_item", "table", "code"]
    # The recorded offsets locate the chunk in the source text.
    source = parsed.text[terms.metadata["start"] : terms.metadata["end"]]
    assert source.startswith("Terms") and source.endswith("```")
    assert "| Rent | 45,000 |" in terms.content  # the table is one unit, never cut

    small = StructureAwareSplitter(max_chunk_size=20).split_parsed(parsed)
    assert all(len(chunk.metadata["block_types"]) == 1 or not chunk.metadata["oversized"] for chunk in small)

    pdf = PyPDFParser().parse(Blob(b"%PDF", "application/pdf", source="lease.pdf"))
    [chunk] = StructureAwareSplitter().split_parsed(pdf, id="doc-1", metadata={"matter": "m"})
    assert chunk.metadata["pages"] == [1, 2] and chunk.metadata["start_page"] == 1
    assert chunk.metadata["methods"] == ["pdf"] and chunk.metadata["matter"] == "m"
    assert chunk.metadata["quality_flags"] == ["paragraph_boundary_heuristic"]
    assert chunk.id == "doc-1#structure_0"


def test_loaders_expose_parse_without_changing_load(tmp_path):
    path = tmp_path / "lease.docx"
    path.write_bytes(docx_bytes())
    loader = DocxLoader(path, metadata={"matter": "m"})
    before = loader.load()
    parsed = loader.parse()
    assert parsed.parser == "docx" and parsed.metadata == {"matter": "m"}
    assert loader.load() == before  # load() is untouched
    note = tmp_path / "note.md"
    note.write_text("# Title\n\nBody")
    assert [e.kind for e in FileLoader(note).parse().elements] == ["heading", "paragraph"]

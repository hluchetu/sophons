import sys
import types

import pytest

from sophons.errors import LoaderError, MissingDependencyError
from sophons.parsers import Blob, OCRParser, ParserDeclined, clean, default_registry


@pytest.fixture
def scans(monkeypatch):
    """Stand-ins for pypdf, pypdfium2 and Pillow driven by the test."""
    state = {"pages": ["Typed first page.", ""], "size": (600, 800), "rendered": [], "closed": 0}

    class Reader:
        def __init__(self, stream):
            self.is_encrypted = False
            self.pages = [types.SimpleNamespace(extract_text=lambda text=text: text) for text in state["pages"]]
            self.trailer = {"/Root": {}}

    class Bitmap:
        def __init__(self, index):
            self.index = index

        def to_pil(self):
            return f"image-of-page-{self.index}"

        def close(self):
            state["closed"] += 1

    class Page:
        def __init__(self, index):
            self.index = index

        def get_size(self):
            return state["size"]

        def render(self, scale):
            state["rendered"].append((self.index, scale))
            return Bitmap(self.index)

        def close(self):
            state["closed"] += 1

    class PdfDocument:
        def __init__(self, data):
            pass

        def __getitem__(self, index):
            return Page(index)

        def close(self):
            state["closed"] += 1

    class Picture:
        size = (100, 100)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def convert(self, mode):
            return "a-photo"

    monkeypatch.setitem(sys.modules, "pypdf", types.SimpleNamespace(PdfReader=Reader))
    monkeypatch.setitem(sys.modules, "pypdfium2", types.SimpleNamespace(PdfDocument=PdfDocument))
    image = types.SimpleNamespace(open=lambda stream: Picture())
    monkeypatch.setitem(sys.modules, "PIL", types.SimpleNamespace(Image=image))
    monkeypatch.setitem(sys.modules, "PIL.Image", image)
    state["Picture"] = Picture
    return state


def reads(image):
    return f"Recognized from {image}."


def test_only_pages_without_text_are_recognized_and_each_page_says_how_it_was_read(scans):
    parsed = OCRParser(recognize=reads).parse(Blob(b"%PDF", "application/pdf", source="scan.pdf"))
    parsed.validate()
    pages = [e for e in parsed.elements if e.kind == "page"]
    assert [(p.page_index, p.method) for p in pages] == [(0, "pdf"), (1, "ocr")]
    assert "ocr_unreviewed" in pages[1].quality_flags and "ocr_unreviewed" not in pages[0].quality_flags
    assert parsed.text == "Typed first page.\n\nRecognized from image-of-page-1."
    assert scans["rendered"] == [(1, 2.0)]  # the typed page was not rendered
    assert scans["closed"] == 3  # bitmap, page and document are all released
    paragraph = [e for e in parsed.elements if e.kind == "paragraph"][1]
    assert paragraph.method == "ocr" and "ocr_unreviewed" in paragraph.quality_flags
    assert clean(parsed).text == parsed.text


def test_images_are_one_recognized_page(scans):
    parsed = OCRParser(recognize=reads).parse(Blob(b"\x89PNG", "image/png"))
    assert parsed.text == "Recognized from a-photo." and parsed.parser == "ocr"
    assert [(e.kind, e.method) for e in parsed.elements] == [("page", "ocr"), ("paragraph", "ocr")]
    scans["Picture"].size = (10_000, 10_000)
    with pytest.raises(LoaderError, match="pixel limit"):
        OCRParser(recognize=reads).parse(Blob(b"\x89PNG", "image/png"))


def test_limits_blank_scans_and_other_types(scans):
    scans["size"] = (6000, 6000)
    with pytest.raises(LoaderError, match="pixel limit"):
        OCRParser(recognize=reads).parse(Blob(b"%PDF", "application/pdf"))
    scans["size"] = (600, 800)
    with pytest.raises(ParserDeclined, match="No readable text"):
        scans["pages"] = ["", ""]
        OCRParser(recognize=lambda image: "  ").parse(Blob(b"%PDF", "application/pdf"))
    with pytest.raises(LoaderError, match="no more than 1 pages"):
        OCRParser(recognize=reads, max_pages=1).parse(Blob(b"%PDF", "application/pdf"))
    with pytest.raises(ParserDeclined, match="only PDFs and images"):
        OCRParser(recognize=reads).parse(Blob(b"hello", "text/plain"))
    with pytest.raises(LoaderError, match="empty"):
        OCRParser(recognize=reads).parse(Blob(b"", "application/pdf"))


def test_registry_falls_through_from_the_text_layer_to_ocr(scans, monkeypatch):
    registry = default_registry()
    for parser in registry.parsers_for("application/pdf"):
        if isinstance(parser, OCRParser):
            monkeypatch.setattr(parser, "_recognize", reads)
    parsed = registry.parse(Blob(b"%PDF", "application/pdf"))
    assert parsed.parser == "ocr" and "Recognized from image-of-page-1." in parsed.text
    scans["pages"] = ["All typed.", "Every page."]
    assert registry.parse(Blob(b"%PDF", "application/pdf")).parser == "pypdf"
    assert [p.name for p in registry.parsers_for("image/jpeg")] == ["ocr"]


def test_missing_ocr_dependencies_are_named(monkeypatch, scans):
    monkeypatch.setitem(sys.modules, "pypdfium2", None)
    with pytest.raises(MissingDependencyError) as missing:
        OCRParser(recognize=reads).parse(Blob(b"%PDF", "application/pdf"))
    assert missing.value.details == {"dependency": "pypdfium2", "extra": "ocr"}
    monkeypatch.setitem(sys.modules, "pytesseract", None)
    with pytest.raises(MissingDependencyError, match="pytesseract"):
        OCRParser().parse(Blob(b"\x89PNG", "image/png"))

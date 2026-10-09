# Structure-aware chunking

Parsing identifies document parts. Splitting decides how those parts fit into
retrieval chunks. These are separate so a future PDF layout parser can reuse the
same chunking policy.

```python
from sophons.documents import Document
from sophons.parsers import TextStructureParser
from sophons.splitters import StructureAwareSplitter

parser = TextStructureParser(numbered_sections=True)
document = Document(
    id="contract",
    content="7. Termination\n7.1 Give written notice.\n7.2 Breach permits termination.",
)

# Inspect what the parser recognized before choosing a chunking policy.
for block in parser.parse(document):
    print(block.kind, block.heading_path, block.content)

splitter = StructureAwareSplitter(parser=parser, max_chunk_size=1000)
for chunk in splitter.split_document(document):
    print(chunk.content, chunk.metadata)
```

The parser recognizes Markdown ATX headings, blank-line-separated paragraphs,
consecutive list-item lines, pipe-delimited table lines, and fenced code blocks.
Numbered-section detection is opt-in. `7.` establishes a top-level boundary;
`7.1` establishes a nested boundary. The complete numbered line is retained as
its heading label because text alone cannot reliably separate a title from a
clause's body. Heading paths are inherited by subsequent blocks. Each block
records its original one-based start and end lines.

The splitter packs consecutive blocks up to a character target, joining them
with blank lines. It starts a new chunk at every heading or change of heading
path. It preserves source metadata and adds heading_path, block_types,
start_line, end_line, chunk_index, parent_id, and oversized. `parent_id` refers
to the input document; it does not automatically create or store section parents.

A block larger than max_chunk_size stays intact and is marked oversized. This
is deliberately a soft target: no hidden cut breaks a table or clause. A heading
may form its own chunk if its following block does not fit. There is no overlap.
Both split_document/split_documents and RAG's split interface are supported.

## Limits and extension points

This parser uses visible text syntax, not font sizes, OCR, or PDF coordinates.
Numbered paragraphs can be mistaken for headings; enable that option only when
it matches your documents. Dates and decimal numbers at line starts may also
look like section numbers. Wrapped list continuations, Setext headings, and
non-pipe tables are not fully recognized. Source text inside blocks is retained,
but whitespace between blocks is normalized when chunks are assembled.

Existing PDFLoader extracts each page separately and DocxLoader flattens text.
Use PDFStructureLoader below for layout-aware PDF conversion.
They do not expose layout/style information to this parser. Page-by-page input
also cannot infer headings carried over from a previous page. For richer file
support, implement StructureParser.parse(document) to return DocumentBlock
objects from a layout-aware loader. Do not flatten structure before parsing it.


## Structure-aware PDF ingestion

Install the optional backend:

```bash
pip install 'sophons[pdf-structure]'
```

```python
from sophons.loaders import PDFStructureLoader
from sophons.splitters import StructureAwareSplitter

# 1. PyMuPDF4LLM examines the PDF layout and converts it to Markdown.
document = PDFStructureLoader("contract.pdf").load()[0]
print(document.content)  # Inspect the inferred headings and tables.

# 2. The text parser identifies the Markdown blocks; the splitter packs them.
chunks = StructureAwareSplitter(max_chunk_size=1000).split(document)
for chunk in chunks:
    print(chunk.metadata["heading_path"], chunk.metadata["pages"])
    print(chunk.content)
```

PDFStructureLoader uses PyMuPDF4LLM's to_markdown(page_chunks=True) API,
then joins the pages into one document so headings can carry across page
boundaries. It records each page's generated Markdown line range. The splitter
uses these ranges to attach pages, page (first page), start_page and end_page
to each chunk. These are source PDF pages, not Markdown line numbers.

The original PDFLoader remains available for simple plain-text extraction.
FileLoader still uses that original loader; select PDFStructureLoader explicitly
for this workflow. PDFStructureLoader does not send documents to a cloud API.

Layout recognition is performed by the external backend and can be imperfect.
Tables crossing page boundaries are not automatically reconstructed. For
numbered clauses not recognized as Markdown headings, supply
TextStructureParser(numbered_sections=True), with the heuristic limitations
noted above. OCR is disabled by default; use use_ocr=True for scanned PDFs
with the backend's OCR dependencies installed. Empty output produces no chunks.
Review PyMuPDF4LLM/PyMuPDF's licensing before distributing an application using
the optional backend; this integration does not change their license terms.

Backend API reference: https://pymupdf.readthedocs.io/en/latest/pymupdf4llm/api.html

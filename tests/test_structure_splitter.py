import pytest

from sophons.documents import Document
from sophons.parsers import DocumentBlock, TextStructureParser
from sophons.splitters import StructureAwareSplitter


def test_parser_tracks_hierarchy_lines_and_code_without_false_headings():
    document = Document(content='# Contract\n\n## Terms\nText\n```python\n# not a heading\n```\n## Payment\nPay now')
    blocks = TextStructureParser().parse(document)
    code = next(block for block in blocks if block.kind == 'code')
    assert code.content == '```python\n# not a heading\n```'
    assert code.heading_path == ('Contract', 'Terms')
    assert (code.start_line, code.end_line) == (5, 7)
    assert blocks[-1].heading_path == ('Contract', 'Payment')


def test_numbered_sections_are_optional_and_preserve_clause_text():
    document = Document(content='7. Termination\n7.1 Either party may terminate.\nNotice must be written.\n7.2 Breach allows termination.')
    splitter = StructureAwareSplitter(parser=TextStructureParser(numbered_sections=True))
    chunks = splitter.split(document)
    assert len(chunks) == 3
    assert chunks[1].metadata['heading_path'] == ['7 Termination', '7.1 Either party may terminate.']
    assert chunks[1].content == '7.1 Either party may terminate.\n\nNotice must be written.'
    assert all(block.kind != 'heading' for block in TextStructureParser().parse(document))


def test_intact_oversized_table_and_metadata():
    table = '| Item | Cost |\n| --- | --- |\n| Service | 500 |'
    document = Document(id='contract', content='# Pricing\n' + table, metadata={'page': 3})
    chunks = StructureAwareSplitter(max_chunk_size=20).split_document(document)
    assert chunks[1].content == table
    assert chunks[1].metadata['oversized'] is True
    assert chunks[1].metadata['page'] == 3
    assert chunks[1].metadata['heading_path'] == ['Pricing']
    assert chunks[1].id == 'contract#structure_1'


def test_custom_parser_and_size_packing():
    class Parser:
        def parse(self, document):
            return [DocumentBlock('abc', 'paragraph'), DocumentBlock('def', 'paragraph')]
    splitter = StructureAwareSplitter(parser=Parser(), max_chunk_size=7)
    assert [chunk.content for chunk in splitter.split(Document(content='ignored'))] == ['abc', 'def']


def test_empty_and_invalid_size():
    assert StructureAwareSplitter().split(Document(content=' \n')) == []
    with pytest.raises(ValueError):
        StructureAwareSplitter(max_chunk_size=0)

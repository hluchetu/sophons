import sys
from types import SimpleNamespace

import pytest

from sophons.errors import MissingDependencyError
from sophons.loaders import PDFStructureLoader
from sophons.splitters import StructureAwareSplitter


def test_missing_pdf_structure_dependency(monkeypatch):
    monkeypatch.setitem(sys.modules, 'pymupdf4llm', None)
    with pytest.raises(MissingDependencyError, match='pdf-structure'):
        PDFStructureLoader('contract.pdf').load()


def test_pdf_layout_to_chunks_keeps_cross_page_heading_and_page_provenance(monkeypatch):
    calls = []
    def convert(path, **kwargs):
        calls.append((path, kwargs))
        return [
            {'text': '# Termination\n\nGive written notice.', 'metadata': {'page_number': 1}},
            {'text': 'Obligations survive.\n\n# Pricing\n\n| Item | Cost |\n| --- | --- |\n| Service | 500 |',
             'metadata': {'page_number': 2}},
        ]
    monkeypatch.setitem(sys.modules, 'pymupdf4llm', SimpleNamespace(to_markdown=convert))
    document = PDFStructureLoader('contract.pdf', id='contract', metadata={'owner': 'team'}).load()[0]
    assert calls == [('contract.pdf', {'page_chunks': True, 'use_ocr': False, 'show_progress': False})]
    chunks = StructureAwareSplitter().split(document)
    assert chunks[0].metadata['heading_path'] == ['Termination']
    assert chunks[0].metadata['pages'] == [1, 2]
    assert chunks[1].metadata['pages'] == [2]
    assert chunks[1].metadata['owner'] == 'team'
    assert 'table' in chunks[1].metadata['block_types']
    assert 'page_line_ranges' not in chunks[0].metadata


def test_empty_pages_and_ocr_option(monkeypatch):
    def convert(path, **kwargs):
        assert kwargs['use_ocr'] is True
        return [{'text': '', 'metadata': {'page_number': 1}}]
    monkeypatch.setitem(sys.modules, 'pymupdf4llm', SimpleNamespace(to_markdown=convert))
    document = PDFStructureLoader('scan.pdf', use_ocr=True).load()[0]
    assert StructureAwareSplitter().split(document) == []


def test_real_pdf_heading_extraction(tmp_path):
    pymupdf = pytest.importorskip('pymupdf')
    pytest.importorskip('pymupdf4llm')
    path = tmp_path / 'contract.pdf'
    with pymupdf.open() as pdf:
        page = pdf.new_page()
        page.insert_text((72, 80), 'Termination', fontsize=20)
        page.insert_text((72, 120), 'Give thirty days written notice.', fontsize=11)
        page.insert_text((72, 160), 'Payment', fontsize=20)
        page.insert_text((72, 200), 'Outstanding invoices remain payable.', fontsize=11)
        pdf.save(path)
    chunks = StructureAwareSplitter().split(PDFStructureLoader(path).load()[0])
    assert [chunk.metadata['heading_path'] for chunk in chunks] == [['Termination'], ['Payment']]
    assert all(chunk.metadata['pages'] == [1] for chunk in chunks)
    assert 'Give thirty days written notice.' in chunks[0].content

from __future__ import annotations

from sophons.loaders.base import AsyncLoader, Loader
from sophons.loaders.directory import DirectoryLoader
from sophons.loaders.docx import DocxLoader
from sophons.loaders.file import FileLoader
from sophons.loaders.pdf import PDFLoader
from sophons.loaders.text import TextLoader
from sophons.loaders.web import WebPageLoader

__all__ = [
    "AsyncLoader",
    "DirectoryLoader",
    "DocxLoader",
    "FileLoader",
    "Loader",
    "PDFLoader",
    "TextLoader",
    "WebPageLoader",
]

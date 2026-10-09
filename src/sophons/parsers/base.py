from __future__ import annotations

from typing import Protocol, runtime_checkable

from sophons.errors import LoaderError, UnsupportedFileTypeError
from sophons.parsers.blob import Blob
from sophons.parsers.elements import ParsedDocument


class ParserDeclined(LoaderError):
    """This parser cannot read this particular file; another one may.

    Raised for a well-formed input the parser is not able to interpret, such as
    a PDF page with no text layer. The registry then tries the next parser.
    """


@runtime_checkable
class Parser(Protocol):
    """Parser contract: bytes in, one structured document out."""

    name: str

    def parse(self, blob: Blob) -> ParsedDocument:
        ...


class ParserRegistry:
    """Ordered parsers per mime type; the first that does not decline wins."""

    def __init__(self) -> None:
        self._parsers: dict[str, list[Parser]] = {}

    def register(self, mime_type: str, parser: Parser, *, first: bool = False) -> None:
        parsers = self._parsers.setdefault(mime_type.strip().lower(), [])
        parsers.insert(0, parser) if first else parsers.append(parser)

    def parsers_for(self, mime_type: str) -> list[Parser]:
        mime_type = mime_type.strip().lower()
        exact = self._parsers.get(mime_type)
        if exact:
            return list(exact)
        family = mime_type.split("/", 1)[0] + "/*"
        return list(self._parsers.get(family, ()))

    def parse(self, blob: Blob) -> ParsedDocument:
        parsers = self.parsers_for(blob.mime_type)
        if not parsers:
            raise UnsupportedFileTypeError(
                f"No parser is registered for {blob.mime_type}",
                details={"mime_type": blob.mime_type},
            )
        declined: list[str] = []
        for parser in parsers:
            try:
                return parser.parse(blob).validate()
            except ParserDeclined as error:
                declined.append(f"{parser.name}: {error}")
        raise ParserDeclined(
            f"No registered parser could read this {blob.mime_type} file",
            details={"declined": declined},
        )

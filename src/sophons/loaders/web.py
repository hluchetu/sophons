from __future__ import annotations

from html.parser import HTMLParser
from typing import Any
from urllib.request import Request, urlopen

from sophons.documents import Document
from sophons.observability import SpanKind, Tracer, maybe_span


class _HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._skip_depth = 0
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in {"script", "style", "noscript"}:
            self._skip_depth += 1
        if tag in {"p", "br", "div", "section", "article", "li", "h1", "h2", "h3"}:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript"} and self._skip_depth:
            self._skip_depth -= 1
        if tag in {"p", "div", "section", "article", "li", "h1", "h2", "h3"}:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        text = " ".join(data.split())
        if text:
            self._parts.append(text)

    def text(self) -> str:
        lines = [line.strip() for line in "".join(self._parts).splitlines()]
        return "\n".join(line for line in lines if line)


class WebPageLoader:
    """Load text from a static web page into a Sophons document."""

    def __init__(
        self,
        url: str,
        *,
        id: str | None = None,
        metadata: dict[str, Any] | None = None,
        timeout: float = 20,
        user_agent: str = "Sophons/0.1",
        tracer: Tracer | None = None,
    ) -> None:
        self.url = url
        self.id = id or url
        self.metadata = dict(metadata or {})
        self.timeout = timeout
        self.user_agent = user_agent
        self._tracer = tracer

    def load(self) -> list[Document]:
        with maybe_span(
            self._tracer,
            "loader.load",
            kind=SpanKind.LOADER,
            loader="web",
            url=self.url,
        ) as span:
            documents = self._load()
            span.set_attribute("document_count", len(documents))
            return documents

    def _load(self) -> list[Document]:
        request = Request(self.url, headers={"User-Agent": self.user_agent})
        with urlopen(request, timeout=self.timeout) as response:
            raw = response.read()
            content_type = response.headers.get("content-type", "text/html")

        html = raw.decode(self._encoding_from_content_type(content_type), errors="replace")
        text = html_to_text(html)
        metadata = {
            "source": self.url,
            "url": self.url,
            "mime_type": content_type.split(";")[0].strip() or "text/html",
        }
        metadata.update(self.metadata)

        return [Document(id=self.id, content=text, metadata=metadata)]

    def lazy_load(self):
        yield from self.load()

    @staticmethod
    def _encoding_from_content_type(content_type: str) -> str:
        for part in content_type.split(";"):
            part = part.strip()
            if part.lower().startswith("charset="):
                return part.split("=", 1)[1]
        return "utf-8"


def html_to_text(html: str) -> str:
    parser = _HTMLTextExtractor()
    parser.feed(html)
    return parser.text()

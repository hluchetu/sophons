from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from sophons.observability.base import Span


class JSONLSink:
    """
    Writes one JSON object per line to a file. For local analysis
    and passing spans to the evals pipeline.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    def emit(self, span: Span) -> None:
        with self._path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(span), default=str) + "\n")

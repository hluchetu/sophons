from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class EvalCase:
    """One test case: a question with whatever ground truth exists for it."""

    id: str
    question: str
    reference: str | None = None
    context: str | None = None
    expected_tools: list[str] | None = None
    expected_tool_calls: list[dict[str, Any]] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class EvalDataset:
    """A named, versioned collection of eval cases.

    Version the dataset like code: scores are only comparable when the
    dataset that produced them is the same.
    """

    name: str
    version: str
    cases: list[EvalCase]

    @classmethod
    def from_json(cls, path: str | Path) -> EvalDataset:
        """
        Load a dataset from a JSON file shaped like::

            {
              "name": "retrieval-support",
              "version": "v1",
              "cases": [
                {"id": "refund-1", "question": "...", "reference": "...",
                 "context": "...", "expected_tools": ["search_docs"]}
              ]
            }
        """
        data = json.loads(Path(path).read_text())
        cases = [
            EvalCase(
                id=str(item["id"]),
                question=str(item["question"]),
                reference=item.get("reference"),
                context=item.get("context"),
                expected_tools=item.get("expected_tools"),
                expected_tool_calls=item.get("expected_tool_calls"),
                metadata=item.get("metadata", {}),
            )
            for item in data["cases"]
        ]
        return cls(
            name=str(data["name"]),
            version=str(data.get("version", "v0")),
            cases=cases,
        )

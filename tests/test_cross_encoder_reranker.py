from __future__ import annotations

import sys
import types

import pytest

from sophons.documents import Document
from sophons.errors import ConfigurationError, ErrorCode, MissingDependencyError
from sophons.integrations.compressors import CrossEncoderReranker


class FakeCrossEncoder:
    calls: list[dict] = []

    def __init__(self, model_name: str, device: str | None = None) -> None:
        self.model_name = model_name
        self.device = device

    def predict(
        self,
        pairs: list[tuple[str, str]],
        *,
        batch_size: int,
        convert_to_numpy: bool,
        show_progress_bar: bool,
    ) -> list[float]:
        self.calls.append(
            {
                "pairs": pairs,
                "batch_size": batch_size,
                "convert_to_numpy": convert_to_numpy,
                "show_progress_bar": show_progress_bar,
            }
        )
        return [0.2, 3.0, 1.0]


@pytest.fixture
def fake_sentence_transformers(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeCrossEncoder.calls = []
    module = types.ModuleType("sentence_transformers")
    setattr(module, "CrossEncoder", FakeCrossEncoder)
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)


@pytest.mark.asyncio
async def test_cross_encoder_reranker_orders_and_scores_documents(
    fake_sentence_transformers: None,
) -> None:
    reranker = CrossEncoderReranker(
        model_name="fake-reranker",
        top_n=2,
        batch_size=4,
        max_chars=5,
        device="cpu",
    )
    documents = [
        Document(id="a", content="alpha document"),
        Document(id="b", content="bravo document"),
        Document(id="c", content="charlie document"),
    ]

    results = await reranker.compress(documents, "refund policy")

    assert [document.id for document in results] == ["b", "c"]
    assert [document.score for document in results] == [3.0, 1.0]
    assert [document.metadata["reranker"] for document in results] == [
        "fake-reranker",
        "fake-reranker",
    ]
    assert FakeCrossEncoder.calls == [
        {
            "pairs": [
                ("refund policy", "alpha"),
                ("refund policy", "bravo"),
                ("refund policy", "charl"),
            ],
            "batch_size": 4,
            "convert_to_numpy": False,
            "show_progress_bar": False,
        }
    ]


@pytest.mark.parametrize(
    ("kwargs", "parameter", "value"),
    [
        ({"top_n": -1}, "top_n", -1),
        ({"batch_size": 0}, "batch_size", 0),
        ({"max_chars": 0}, "max_chars", 0),
    ],
)
def test_cross_encoder_reranker_rejects_invalid_parameters(
    fake_sentence_transformers: None,
    kwargs: dict[str, int],
    parameter: str,
    value: int,
) -> None:
    with pytest.raises(ConfigurationError, match=parameter) as error:
        CrossEncoderReranker(**kwargs)

    assert error.value.error_code == ErrorCode.CONFIGURATION_ERROR
    assert error.value.details == {"parameter": parameter, "value": value}


def test_cross_encoder_reranker_reports_missing_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delitem(sys.modules, "sentence_transformers", raising=False)

    class MissingSentenceTransformers:
        def find_spec(self, fullname: str, path: object = None, target: object = None):
            if fullname == "sentence_transformers":
                return None
            return None

    monkeypatch.setattr(sys, "meta_path", [MissingSentenceTransformers()])

    with pytest.raises(MissingDependencyError, match="sentence-transformers") as error:
        CrossEncoderReranker()

    assert error.value.error_code == ErrorCode.MISSING_DEPENDENCY

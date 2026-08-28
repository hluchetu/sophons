from __future__ import annotations

import asyncio
import subprocess
import sys
import types
from types import SimpleNamespace

import pytest

from sophons.embeddings import AsyncEmbeddingModel, EmbeddingModel


def test_public_api_is_importable_without_provider_sdks() -> None:
    code = """
import sys
import sophons.embeddings as embeddings

assert "openai" not in sys.modules
assert "sentence_transformers" not in sys.modules
assert "OpenAIEmbeddings" in embeddings.__all__
assert "AsyncOpenAIEmbeddings" in dir(embeddings)

from sophons.embeddings import OpenAIEmbeddings

assert OpenAIEmbeddings.__name__ == "OpenAIEmbeddings"
assert "openai" not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_unknown_public_name_raises_attribute_error() -> None:
    from sophons import embeddings

    with pytest.raises(AttributeError, match="MissingEmbedder"):
        _ = embeddings.MissingEmbedder


def test_openai_implementations_match_protocols() -> None:
    from sophons.embeddings import AsyncOpenAIEmbeddings, OpenAIEmbeddings

    sync_embedder = OpenAIEmbeddings.__new__(OpenAIEmbeddings)
    async_embedder = AsyncOpenAIEmbeddings.__new__(AsyncOpenAIEmbeddings)

    assert isinstance(sync_embedder, EmbeddingModel)
    assert isinstance(async_embedder, AsyncEmbeddingModel)


def test_sync_openai_empty_batch_does_not_require_a_client() -> None:
    from sophons.embeddings import OpenAIEmbeddings

    embedder = OpenAIEmbeddings.__new__(OpenAIEmbeddings)
    assert embedder.embed_documents([]) == []


def test_async_openai_empty_batch_does_not_require_a_client() -> None:
    from sophons.embeddings import AsyncOpenAIEmbeddings

    embedder = AsyncOpenAIEmbeddings.__new__(AsyncOpenAIEmbeddings)
    assert asyncio.run(embedder.embed_documents([])) == []


def test_sync_openai_forwards_configuration_and_preserves_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    class FakeEmbeddings:
        def create(self, **kwargs: object) -> SimpleNamespace:
            calls.append(kwargs)
            return SimpleNamespace(
                data=[
                    SimpleNamespace(embedding=[1.0, 0.0]),
                    SimpleNamespace(embedding=[0.0, 1.0]),
                ]
            )

    class FakeOpenAI:
        def __init__(self, *, api_key: str) -> None:
            assert api_key == "test-key"
            self.embeddings = FakeEmbeddings()

    module = types.ModuleType("openai")
    module.OpenAI = FakeOpenAI  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openai", module)

    from sophons.embeddings import OpenAIEmbeddings

    embedder = OpenAIEmbeddings(
        api_key="test-key",
        model="test-model",
        dimensions=2,
    )
    vectors = embedder.embed_documents(["first", "second"])

    assert vectors == [[1.0, 0.0], [0.0, 1.0]]
    assert calls == [
        {
            "input": ["first", "second"],
            "model": "test-model",
            "dimensions": 2,
        }
    ]


def test_sentence_transformers_forwards_model_and_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[object, bool]] = []

    class Encoded:
        def __init__(self, value: object) -> None:
            self._value = value

        def tolist(self) -> object:
            return self._value

    class FakeSentenceTransformer:
        def __init__(self, model: str, *, device: str) -> None:
            assert model == "test-model"
            assert device == "cpu"

        def encode(self, value: object, *, normalize_embeddings: bool) -> Encoded:
            calls.append((value, normalize_embeddings))
            if isinstance(value, str):
                return Encoded([1.0, 2.0])
            return Encoded([[1.0, 2.0], [3.0, 4.0]])

    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = FakeSentenceTransformer  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)

    from sophons.embeddings import SentenceTransformerEmbeddings

    embedder = SentenceTransformerEmbeddings(
        model="test-model",
        device="cpu",
        normalize=True,
    )
    assert embedder.embed_query("query") == [1.0, 2.0]
    assert embedder.embed_documents(["first", "second"]) == [
        [1.0, 2.0],
        [3.0, 4.0],
    ]
    assert calls == [
        ("query", True),
        (["first", "second"], True),
    ]


def test_model_integrations_are_lazy_and_chat_only() -> None:
    code = """
import sys
import sophons.integrations.models as models

assert set(models.__all__) == {
    "AnthropicModel",
    "BedrockModel",
    "DeepSeekModel",
    "ModelSettings",
    "OllamaModel",
}
assert "openai" not in sys.modules
assert "anthropic" not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_ollama_settings_do_not_require_a_provider_sdk() -> None:
    from sophons.integrations.models import ModelSettings, OllamaModel

    settings = ModelSettings(max_tokens=512)
    model = OllamaModel(model="test", settings=settings)

    assert model._settings.max_tokens == 512

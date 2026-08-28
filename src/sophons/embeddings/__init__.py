from __future__ import annotations

from importlib import import_module
from typing import Any

from sophons.models.embeddings import AsyncEmbeddingModel, EmbeddingModel, Vector

__all__ = [
    "AsyncEmbeddingModel",
    "AsyncOpenAIEmbeddings",
    "EmbeddingModel",
    "OpenAIEmbeddings",
    "SentenceTransformerEmbeddings",
    "Vector",
]

_LAZY_IMPORTS = {
    "AsyncOpenAIEmbeddings": (
        "sophons.integrations.embeddings.openai",
        "AsyncOpenAIEmbeddings",
    ),
    "OpenAIEmbeddings": (
        "sophons.integrations.embeddings.openai",
        "OpenAIEmbeddings",
    ),
    "SentenceTransformerEmbeddings": (
        "sophons.integrations.embeddings.sentence_transformers",
        "SentenceTransformerEmbeddings",
    ),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute_name = _LAZY_IMPORTS[name]
    except KeyError as exc:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        ) from exc

    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))

from __future__ import annotations

from importlib import import_module
from typing import Any

from sophons.retrieval.base import VectorStore

__all__ = [
    "ChromaVectorStore",
    "InMemoryVectorStore",
    "NumPyVectorStore",
    "VectorStore",
]

_LAZY_IMPORTS = {
    "ChromaVectorStore": (
        "sophons.integrations.vector_stores.chroma",
        "ChromaVectorStore",
    ),
    "InMemoryVectorStore": (
        "sophons.integrations.vector_stores.in_memory",
        "InMemoryVectorStore",
    ),
    "NumPyVectorStore": (
        "sophons.integrations.vector_stores.numpy",
        "NumPyVectorStore",
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

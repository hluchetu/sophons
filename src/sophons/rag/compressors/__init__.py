from __future__ import annotations

from importlib import import_module
from typing import Any

from sophons.rag.compressors.interface import DocumentCompressor

__all__ = ["DocumentCompressor"]

_LAZY_IMPORTS = {
    "CrossEncoderReranker": (
        "sophons.integrations.compressors.sentence_transformers",
        "CrossEncoderReranker",
    ),
}


def __getattr__(name: str) -> Any:
    """Load optional compressor integrations only when accessed."""
    try:
        module_name, attribute_name = _LAZY_IMPORTS[name]
    except KeyError as exc:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        ) from exc

    module = import_module(module_name)
    value = getattr(module, attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Return eagerly and lazily exposed public names."""
    return sorted(set(globals()) | set(__all__) | set(_LAZY_IMPORTS))

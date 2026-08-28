from __future__ import annotations

from importlib import import_module
from typing import Any

from sophons.models.chat import AsyncChatModel, ChatModel
from sophons.models.embeddings import AsyncEmbeddingModel, EmbeddingModel, Vector
from sophons.models.messages import Message

__all__ = [
    "AsyncChatModel",
    "AsyncEmbeddingModel",
    "ChatModel",
    "EmbeddingModel",
    "Message",
    "Vector",
]

_LAZY_IMPORTS = {
    "AnthropicModel": (
        "sophons.integrations.models.anthropic",
        "AnthropicModel",
    ),
    "BedrockModel": (
        "sophons.integrations.models.bedrock",
        "BedrockModel",
    ),
    "DeepSeekModel": (
        "sophons.integrations.models.deepseek",
        "DeepSeekModel",
    ),
    "ModelSettings": (
        "sophons.integrations.models.settings",
        "ModelSettings",
    ),
    "OllamaModel": (
        "sophons.integrations.models.ollama",
        "OllamaModel",
    ),
}

__all__ += list(_LAZY_IMPORTS)


def __getattr__(name: str) -> Any:
    """Load optional model integrations only when accessed."""
    try:
        module_name, attribute_name = _LAZY_IMPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    module = import_module(module_name)
    value = getattr(module, attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Return eagerly and lazily exposed public names."""
    return sorted(set(globals()) | set(__all__) | set(_LAZY_IMPORTS))

from __future__ import annotations

from typing import TYPE_CHECKING

import lazy_loader as lazy

if TYPE_CHECKING:
    from sophons.integrations.models.anthropic import AnthropicModel
    from sophons.integrations.models.bedrock import BedrockModel
    from sophons.integrations.models.deepseek import DeepSeekModel
    from sophons.integrations.models.ollama import OllamaModel
    from sophons.integrations.models.settings import ModelSettings

    __all__ = [
        "AnthropicModel",
        "BedrockModel",
        "DeepSeekModel",
        "ModelSettings",
        "OllamaModel",
    ]
else:
    __getattr__, __dir__, __all__ = lazy.attach(
        __name__,
        submod_attrs={
            "anthropic": ["AnthropicModel"],
            "bedrock": ["BedrockModel"],
            "deepseek": ["DeepSeekModel"],
            "ollama": ["OllamaModel"],
            "settings": ["ModelSettings"],
        },
    )

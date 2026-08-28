from __future__ import annotations

import subprocess
import sys
import types

import pytest

from sophons.models import ChatModel


def test_public_model_api_is_importable_without_provider_sdks() -> None:
    code = """
import sys
import sophons.models as models

assert "openai" not in sys.modules
assert "anthropic" not in sys.modules
assert "boto3" not in sys.modules
assert "DeepSeekModel" in models.__all__
assert "BedrockModel" in models.__all__
assert "AnthropicModel" in dir(models)

from sophons.models import DeepSeekModel

assert DeepSeekModel.__name__ == "DeepSeekModel"
assert "openai" not in sys.modules
assert "boto3" not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_unknown_public_model_name_raises_attribute_error() -> None:
    from sophons import models

    with pytest.raises(AttributeError, match="MissingModel"):
        _ = models.MissingModel


def test_deepseek_model_matches_chat_protocol_without_construction() -> None:
    from sophons.models import DeepSeekModel

    model = DeepSeekModel.__new__(DeepSeekModel)
    assert isinstance(model, ChatModel)


def test_bedrock_model_matches_chat_protocol_without_construction() -> None:
    from sophons.models import BedrockModel

    model = BedrockModel.__new__(BedrockModel)
    assert isinstance(model, ChatModel)


def test_deepseek_constructs_openai_compatible_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received: dict[str, str] = {}

    class FakeOpenAI:
        def __init__(self, *, api_key: str, base_url: str) -> None:
            received.update(api_key=api_key, base_url=base_url)

    module = types.ModuleType("openai")
    module.OpenAI = FakeOpenAI  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openai", module)

    from sophons.models import DeepSeekModel

    model = DeepSeekModel(
        model="deepseek-chat",
        api_key="test-key",
    )

    assert model.model == "deepseek-chat"
    assert received == {
        "api_key": "test-key",
        "base_url": "https://api.deepseek.com/v1",
    }


def test_legacy_integration_import_remains_available() -> None:
    from sophons.integrations.models import DeepSeekModel as IntegrationModel
    from sophons.models import DeepSeekModel as PublicModel

    assert IntegrationModel is PublicModel


def test_legacy_bedrock_integration_import_remains_available() -> None:
    from sophons.integrations.models import BedrockModel as IntegrationModel
    from sophons.models import BedrockModel as PublicModel

    assert IntegrationModel is PublicModel


def test_structured_output_tool_is_public() -> None:
    from sophons.agents import OutputTool

    assert OutputTool.__name__ == "OutputTool"

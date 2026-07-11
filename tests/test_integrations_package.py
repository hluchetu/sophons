from __future__ import annotations

import sophons.integrations.agent_sdks
import sophons.integrations.mcp
import sophons.integrations.models
import sophons.integrations.sources
import sophons.integrations.storage
import sophons.integrations.vector_stores


def test_integration_subpackages_import() -> None:
    assert "AnthropicModel" in sophons.integrations.models.__all__
    assert "DeepSeekModel" in sophons.integrations.models.__all__
    assert "ChromaVectorStore" in sophons.integrations.vector_stores.__all__
    assert sophons.integrations.agent_sdks.__all__ == []
    assert sophons.integrations.storage.__all__ == []
    assert sophons.integrations.sources.__all__ == []
    assert sophons.integrations.mcp.__all__ == []

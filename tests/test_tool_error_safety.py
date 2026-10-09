"""Raw tool exceptions must not enter model-visible or runtime outputs."""
import logging
import pytest
from sophons.agents import Agent
from sophons.models import Message

SECRET = "SECRET_TOKEN=synthetic-private-detail"

class Model:
    def __init__(self):
        self.calls = 0
        self.seen = []

    def invoke(self, messages, tools=None):
        self.seen.extend(str(message) for message in messages)
        self.calls += 1
        if self.calls == 1:
            return Message(role="assistant", content="", metadata={"tool_calls": [
                {"tool_use_id": "original-id", "name": "broken", "input": {}}
            ]})
        return Message(role="assistant", content="Unable to complete the operation.")

class Broken:
    name = "broken"
    description = "Synthetic failing tool"
    def schema(self):
        return {"type": "object", "properties": {}}
    def call(self, args):
        raise RuntimeError(SECRET)

class AsyncBroken(Broken):
    async def call(self, args):
        raise RuntimeError(SECRET)

class Unserializable(Broken):
    def call(self, args):
        class Private:
            def __repr__(self):
                return SECRET
        return {"value": Private()}

@pytest.mark.parametrize("tool", [Broken(), AsyncBroken(), Unserializable()])
def test_tool_failure_is_sanitized(tool, caplog):
    caplog.set_level(logging.DEBUG)
    model = Model()
    result = Agent(model=model, tools=[tool]).run_sync("Try tool")
    failure = result.tool_results[0]
    assert failure.tool_use_id == "original-id"
    assert failure.status == "error"
    assert failure.content == "Tool execution failed. The operation could not be completed."
    assert result.metrics.tool_calls == 1
    assert SECRET not in str(result) + str(model.seen) + caplog.text

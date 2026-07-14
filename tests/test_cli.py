from __future__ import annotations

import sophons.cli as cli
from sophons.agents.responses import AgentMetrics


class FakeResult:
    message = "hello"
    metrics = AgentMetrics(steps=2, model_calls=2, tool_calls=1)


class FakeAgent:
    def __init__(self) -> None:
        self.asked: list[str] = []

    def run_sync(self, question: str, *, session_id=None) -> FakeResult:
        self.asked.append(question)
        return FakeResult()


def test_chat_with_agent_wires_answer_and_metrics(monkeypatch) -> None:
    captured: dict = {}

    def fake_chat(*, title, subtitle="", answer, history_name="x"):
        captured["title"] = title
        captured["reply"] = answer("what is up?")

    monkeypatch.setattr(cli, "chat", fake_chat)
    agent = FakeAgent()

    cli.chat_with_agent(agent, title="Test Agent")

    assert agent.asked == ["what is up?"]
    assert captured["title"] == "Test Agent"
    text, footer = captured["reply"]
    assert text == "hello"
    assert footer == "steps=2  model_calls=2  tool_calls=1"

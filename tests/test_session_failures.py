"""Persistence failure is an explicit outcome, never a fresh/successful turn."""
import pytest
from sophons.agents import Agent, FileSessionManager, InMemorySessionManager, AgentFinished, AgentFailed, SessionPersistenceError
from sophons.models import Message


class Model:
    def __init__(self):
        self.calls = 0

    def invoke(self, messages, tools=None):
        self.calls += 1
        return Message(role="assistant", content="Candidate answer")


class BrokenSessions:
    def __init__(self, operation):
        self.operation = operation
        self.saves = 0

    async def load(self, session_id):
        if self.operation == "load":
            raise OSError("SECRET_DATABASE_CONNECTION_DETAILS")
        return []

    async def save(self, session_id, messages):
        self.saves += 1
        raise OSError("SECRET_DATABASE_CONNECTION_DETAILS")


@pytest.mark.asyncio
async def test_failed_load_stops_before_model_work(caplog):
    model, sessions = Model(), BrokenSessions("load")
    result = await Agent(model=model, session_manager=sessions).run("Question", session_id="owned-session")
    assert model.calls == 0 and sessions.saves == 0
    assert not result.success and result.stop_reason.value == "session_error"
    assert "SECRET_DATABASE" not in result.message + (result.error or "") + caplog.text


@pytest.mark.asyncio
async def test_failed_save_does_not_return_a_successful_answer(caplog):
    model, sessions = Model(), BrokenSessions("save")
    result = await Agent(model=model, session_manager=sessions).run("Question", session_id="owned-session")
    assert model.calls == 1 and sessions.saves == 1
    assert not result.success and result.stop_reason.value == "session_error"
    assert result.output is None and "Candidate answer" not in result.message
    assert "SECRET_DATABASE" not in result.message + (result.error or "") + caplog.text


@pytest.mark.asyncio
async def test_corrupt_file_is_not_treated_as_an_empty_conversation(tmp_path):
    sessions = FileSessionManager(tmp_path)
    (tmp_path / "owned-session.json").write_text("broken JSON")
    model = Model()
    result = await Agent(model=model, session_manager=sessions).run("Question", session_id="owned-session")
    assert model.calls == 0
    assert not result.success and result.stop_reason.value == "session_error"
    assert (tmp_path / "owned-session.json").read_text() == "broken JSON"


@pytest.mark.asyncio
async def test_missing_file_is_a_valid_new_session(tmp_path):
    sessions = FileSessionManager(tmp_path)
    assert await sessions.load("new-session") == []
    result = await Agent(model=Model(), session_manager=sessions).run("Question", session_id="new-session")
    assert result.success
    assert len(await sessions.load("new-session")) == 2


@pytest.mark.asyncio
async def test_failed_atomic_replace_preserves_previous_history(tmp_path, monkeypatch):
    import sophons.agents.session as module
    sessions = FileSessionManager(tmp_path)
    await sessions.save("owned-session", [Message(role="user", content="Previous history")])
    original = (tmp_path / "owned-session.json").read_bytes()

    def reject_replace(*args):
        raise PermissionError("SECRET_STORAGE_DETAIL")
    monkeypatch.setattr(module.os, "replace", reject_replace)
    with pytest.raises(SessionPersistenceError) as error:
        await sessions.save("owned-session", [Message(role="user", content="Replacement")])
    assert error.value.operation == "save"
    assert "SECRET_STORAGE" not in str(error.value)
    assert (tmp_path / "owned-session.json").read_bytes() == original
    assert not list(tmp_path.glob(".sophons-session-*.tmp"))


@pytest.mark.asyncio
async def test_serialization_failure_does_not_damage_old_file(tmp_path):
    sessions = FileSessionManager(tmp_path)
    await sessions.save("owned-session", [Message(role="user", content="Previous")])
    original = (tmp_path / "owned-session.json").read_bytes()
    with pytest.raises(SessionPersistenceError):
        await sessions.save("owned-session", [Message(role="user", content="New", metadata={"bad": object()})])
    assert (tmp_path / "owned-session.json").read_bytes() == original


@pytest.mark.asyncio
async def test_failed_save_emits_failure_not_completion_and_skips_memory(monkeypatch):
    finished, failed, memory_writes = [], [], []
    agent = Agent(model=Model(), session_manager=BrokenSessions("save"))
    def on_finished(event: AgentFinished):
        finished.append(event)
    def on_failed(event: AgentFailed):
        failed.append(event)
    async def add_memory(*args):
        memory_writes.append(True)
    agent.add_hook(on_finished)
    agent.add_hook(on_failed)
    monkeypatch.setattr(agent, "_add_memory", add_memory)
    result = await agent.run("Question", session_id="owned-session")
    assert not result.success and not finished and not memory_writes
    assert len(failed) == 1 and isinstance(failed[0].error, SessionPersistenceError)


@pytest.mark.asyncio
async def test_completion_event_happens_only_after_successful_save():
    order = []
    class Sessions(InMemorySessionManager):
        async def save(self, session_id, messages):
            order.append("save")
            await super().save(session_id, messages)
    agent = Agent(model=Model(), session_manager=Sessions())
    def finished(event: AgentFinished):
        order.append("finished")
    agent.add_hook(finished)
    result = await agent.run("Question", session_id="owned-session")
    assert result.success and order == ["save", "finished"]


@pytest.mark.asyncio
async def test_no_session_id_skips_persistence():
    sessions = BrokenSessions("load")
    result = await Agent(model=Model(), session_manager=sessions).run("Question")
    assert result.success and sessions.saves == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("body", ['{}', 'null', '[{"role": "unknown", "content": "bad"}]', '[{"role": "user", "content": 12}]'])
async def test_malformed_stored_records_raise_explicit_error(tmp_path, body):
    (tmp_path / "owned-session.json").write_text(body)
    with pytest.raises(SessionPersistenceError) as error:
        await FileSessionManager(tmp_path).load("owned-session")
    assert error.value.operation == "load"


@pytest.mark.asyncio
async def test_in_memory_history_still_round_trips():
    sessions = InMemorySessionManager()
    message = Message(role="user", content="Synthetic history", id="message-1")
    await sessions.save("session-1", [message])
    assert await sessions.load("session-1") == [message]
    await sessions.delete("session-1")
    assert not await sessions.exists("session-1")


@pytest.mark.asyncio
async def test_unavailable_store_is_not_mistaken_for_a_new_session(tmp_path):
    directory = tmp_path / "storage"
    sessions = FileSessionManager(directory)
    directory.rmdir()
    model = Model()
    result = await Agent(model=model, session_manager=sessions).run("Question", session_id="new-session")
    assert model.calls == 0 and result.stop_reason.value == "session_error"
    with pytest.raises(SessionPersistenceError):
        await sessions.exists("new-session")
    with pytest.raises(SessionPersistenceError):
        await sessions.delete("new-session")

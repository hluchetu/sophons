"""Run budgets and deadlines must surround retries and in-flight work."""
import asyncio
import threading

import pytest
from sophons.agents import Agent, RunLimits, exponential_backoff
from sophons.models import Message


class Unavailable(Exception):
    status_code = 503


class FailingModel:
    def __init__(self):
        self.calls = 0

    async def invoke(self, messages, tools=None):
        self.calls += 1
        raise Unavailable("Synthetic unavailable response")


@pytest.mark.asyncio
async def test_retry_attempts_cannot_exceed_shared_model_budget():
    model = FailingModel()
    result = await Agent(model=model, limits=RunLimits(max_model_calls=2),
        retry_strategy=exponential_backoff(max_attempts=6, initial_delay=0, max_delay=0, jitter=False)).run("Question")
    assert model.calls == 2
    assert result.metrics.model_calls == 2
    assert result.stop_reason.value == "max_model_calls"
    assert not result.success


@pytest.mark.asyncio
async def test_deadline_cancels_an_in_flight_async_model():
    class SlowModel:
        cancelled = False

        async def invoke(self, messages, tools=None):
            try:
                await asyncio.sleep(0.2)
                return Message(role="assistant", content="Too late")
            except asyncio.CancelledError:
                self.cancelled = True
                raise

    model = SlowModel()
    result = await Agent(model=model, limits=RunLimits(max_runtime_seconds=0.03)).run("Question")
    assert model.cancelled
    assert not result.success and result.stop_reason.value == "max_runtime"


@pytest.mark.asyncio
async def test_pre_cancelled_invocation_does_not_call_model():
    signal = threading.Event()
    signal.set()
    model = FailingModel()
    result = await Agent(model=model).run("Question", cancel_signal=signal)
    assert model.calls == 0
    assert result.stop_reason.value == "cancelled" and not result.success


@pytest.mark.asyncio
async def test_success_after_retry_counts_both_attempts():
    class EventuallyWorks(FailingModel):
        async def invoke(self, messages, tools=None):
            self.calls += 1
            if self.calls == 1:
                raise Unavailable("Transient")
            return Message(role="assistant", content="Done")
    model = EventuallyWorks()
    result = await Agent(model=model, limits=RunLimits(max_model_calls=2),
        retry_strategy=exponential_backoff(initial_delay=0, max_delay=0, jitter=False)).run("Question")
    assert result.success and model.calls == result.metrics.model_calls == 2


@pytest.mark.asyncio
async def test_backoff_that_cannot_fit_deadline_does_not_start_next_attempt():
    model = FailingModel()
    result = await Agent(model=model, limits=RunLimits(max_runtime_seconds=0.1),
        retry_strategy=exponential_backoff(initial_delay=10, max_delay=10, jitter=False)).run("Question")
    assert model.calls == 1 and result.metrics.model_calls == 1
    assert result.stop_reason.value == "max_runtime"


@pytest.mark.asyncio
async def test_permanent_failure_is_not_retried():
    class PermanentModel(FailingModel):
        async def invoke(self, messages, tools=None):
            self.calls += 1
            raise ValueError("Synthetic permanent failure")
    model = PermanentModel()
    result = await Agent(model=model).run("Question")
    assert model.calls == result.metrics.model_calls == 1
    assert not result.success and result.stop_reason.value == "error"


@pytest.mark.asyncio
async def test_external_signal_cancels_in_flight_work_and_is_not_cleared():
    started = asyncio.Event()
    class WaitingModel:
        cancelled = False
        async def invoke(self, messages, tools=None):
            started.set()
            try:
                await asyncio.sleep(1)
            except asyncio.CancelledError:
                self.cancelled = True
                raise
    signal = threading.Event()
    model = WaitingModel()
    task = asyncio.create_task(Agent(model=model).run("Question", cancel_signal=signal))
    await started.wait()
    signal.set()
    result = await task
    assert model.cancelled and signal.is_set()
    assert result.stop_reason.value == "cancelled" and not result.success


@pytest.mark.asyncio
async def test_asyncio_task_cancellation_propagates_to_caller():
    started = asyncio.Event()
    class WaitingModel:
        async def invoke(self, messages, tools=None):
            started.set()
            await asyncio.sleep(1)
    task = asyncio.create_task(Agent(model=WaitingModel()).run("Question"))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


def test_blocking_sync_model_does_not_hold_up_deadline_or_publish_late_answer():
    release, finished = threading.Event(), threading.Event()
    class BlockingModel:
        def invoke(self, messages, tools=None):
            release.wait(2)
            finished.set()
            return Message(role="assistant", content="Late result")
    try:
        result = Agent(model=BlockingModel(), limits=RunLimits(max_runtime_seconds=0.03)).run_sync("Question")
        assert not finished.is_set()
        assert not result.success and result.stop_reason.value == "max_runtime"
        assert "Late result" not in result.message
    finally:
        release.set()
        finished.wait(1)


@pytest.mark.asyncio
async def test_deadline_cancels_an_async_tool_before_completion():
    class ToolModel:
        def invoke(self, messages, tools=None):
            return Message(role="assistant", content="", metadata={"tool_calls": [
                {"tool_use_id": "slow", "name": "slow", "input": {}}
            ]})
    class SlowTool:
        name, description, args_schema = "slow", "Synthetic slow tool", {"type": "object"}
        cancelled = False
        async def call(self, args):
            try:
                await asyncio.sleep(0.2)
            except asyncio.CancelledError:
                self.cancelled = True
                raise
    tool = SlowTool()
    result = await Agent(model=ToolModel(), tools=[tool], limits=RunLimits(max_runtime_seconds=0.03)).run("Question")
    assert tool.cancelled
    assert result.stop_reason.value == "max_runtime" and not result.success


@pytest.mark.asyncio
async def test_context_recovery_attempt_counts_toward_model_budget():
    from sophons.agents import SlidingWindowManager
    from sophons.errors import ContextWindowOverflowError
    class OverflowModel(FailingModel):
        async def invoke(self, messages, tools=None):
            self.calls += 1
            raise ContextWindowOverflowError("Synthetic context overflow")
    model = OverflowModel()
    result = await Agent(model=model, limits=RunLimits(max_model_calls=1),
        conversation_manager=SlidingWindowManager(max_messages=10)).run("Question")
    assert model.calls == result.metrics.model_calls == 1
    assert result.stop_reason.value == "max_model_calls"


@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_invalid_retry_attempt_ceiling_is_rejected(value):
    from sophons.agents import RetryStrategy
    with pytest.raises(ValueError, match="positive integer"):
        RetryStrategy(max_attempts=value)


@pytest.mark.parametrize("value", [float("inf"), float("nan"), -1, True])
def test_invalid_runtime_deadline_is_rejected(value):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        RunLimits(max_runtime_seconds=value)


@pytest.mark.asyncio
async def test_cancel_during_backoff_never_starts_another_attempt():
    attempted = asyncio.Event()
    class FailedOnce(FailingModel):
        async def invoke(self, messages, tools=None):
            self.calls += 1
            attempted.set()
            raise Unavailable("Synthetic failure")
    model, signal = FailedOnce(), threading.Event()
    agent = Agent(model=model, retry_strategy=exponential_backoff(initial_delay=1, max_delay=1, jitter=False))
    task = asyncio.create_task(agent.run("Question", cancel_signal=signal))
    await attempted.wait()
    signal.set()
    result = await task
    assert model.calls == result.metrics.model_calls == 1
    assert result.stop_reason.value == "cancelled"


def test_model_budget_is_fresh_for_each_run():
    class FastModel:
        def invoke(self, messages, tools=None):
            return Message(role="assistant", content="Done")
    agent = Agent(model=FastModel(), limits=RunLimits(max_model_calls=1))
    first, second = agent.run_sync("First"), agent.run_sync("Second")
    assert first.success and second.success
    assert first.metrics.model_calls == second.metrics.model_calls == 1

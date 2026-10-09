"""Invocation-owned deadlines and cancellation, separate from retry policy.

Design reference: Strands lifecycle/retry separation; implemented for Sophons'
interfaces. No Strands source is vendored here.
"""
from __future__ import annotations

import asyncio
import contextvars
import inspect
import threading
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from sophons.agents.responses import StopReason
from sophons.agents.state import RunLimits, RunState

T = TypeVar("T")


class RunStopped(Exception):
    """Terminal control outcome: retry policies must never override it."""

    def __init__(self, reason: StopReason):
        self.reason = reason
        super().__init__(f"Run stopped: {reason.value}")


class RunControl:
    def __init__(self, state: RunState, limits: RunLimits,
                 cancel_signal: threading.Event | None = None):
        self.state, self.limits, self.cancel_signal = state, limits, cancel_signal

    def remaining_seconds(self) -> float:
        return self.limits.max_runtime_seconds - self.state.elapsed_seconds()

    def check(self) -> None:
        if self.cancel_signal is not None and self.cancel_signal.is_set():
            raise RunStopped(StopReason.CANCELLED)
        if self.remaining_seconds() <= 0:
            raise RunStopped(StopReason.MAX_RUNTIME)

    def reserve_model_attempt(self) -> None:
        self.check()
        if not self.state.reserve_model_call(self.limits):
            raise RunStopped(StopReason.MAX_MODEL_CALLS)

    def before_retry_sleep(self, delay: float) -> None:
        self.check()
        if self.state.model_call_count >= self.limits.max_model_calls:
            raise RunStopped(StopReason.MAX_MODEL_CALLS)
        if delay >= self.remaining_seconds():
            raise RunStopped(StopReason.MAX_RUNTIME)

    async def wait(self, operation: Callable[[], Awaitable[T]]) -> T:
        """Bound waiting; async operations must cooperate with cancellation."""
        self.check()
        task = asyncio.ensure_future(operation())
        try:
            while True:
                self.check()
                interval = self.remaining_seconds()
                if self.cancel_signal is not None:
                    interval = min(interval, 0.02)
                done, _ = await asyncio.wait({task}, timeout=interval)
                self.check()
                if done:
                    return await task
        finally:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)


async def call_callable(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Run async callables directly and isolate blocking sync work in a daemon.

    Stopping the wait cannot kill a Python thread or undo tool side effects.
    Provider/tool timeouts or process isolation are still required for cleanup.
    Late thread results are discarded; they cannot complete a cancelled future.
    """
    if inspect.iscoroutinefunction(fn):
        return await fn(*args, **kwargs)
    loop = asyncio.get_running_loop()
    future = loop.create_future()
    context = contextvars.copy_context()

    def complete(outcome):
        if future.done():
            return
        success, value = outcome
        if success:
            future.set_result(value)
        else:
            future.set_exception(value)

    def worker():
        try:
            outcome = (True, context.run(fn, *args, **kwargs))
        except BaseException as error:
            outcome = (False, error)
        try:
            loop.call_soon_threadsafe(complete, outcome)
        except RuntimeError:
            pass  # The invocation's loop has closed; discard the late result.

    threading.Thread(target=worker, daemon=True).start()
    value = await future
    return await value if inspect.isawaitable(value) else value

from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar, cast

logger = logging.getLogger(__name__)

T = TypeVar("T")

# ---------------------------------------------------------------------------
# Defaults  (match OpenAI Agents SDK defaults — battle-tested values)
# ---------------------------------------------------------------------------

DEFAULT_MAX_ATTEMPTS: int = 6
DEFAULT_INITIAL_DELAY: float = 0.25
DEFAULT_MAX_DELAY: float = 2.0
DEFAULT_MULTIPLIER: float = 2.0
DEFAULT_JITTER: bool = True


# ---------------------------------------------------------------------------
# Retry context — what the policy sees when deciding
# ---------------------------------------------------------------------------


@dataclass
class RetryContext:
    """
    Information passed to a retry policy so it can decide whether to retry.

    Attributes:
        error:      The exception that was raised.
        attempt:    Which attempt just failed (1 = first attempt).
        max_attempts: The configured ceiling.
    """

    error: Exception
    attempt: int
    max_attempts: int


# ---------------------------------------------------------------------------
# Retry decision — what the policy returns
# ---------------------------------------------------------------------------


@dataclass
class RetryDecision:
    """
    The policy's verdict for a failed attempt.

    Attributes:
        retry:  Whether to retry at all.
        delay:  How long to wait before the next attempt.
                If None the executor calculates the backoff delay.
        reason: Human-readable explanation (used in logs).
    """

    retry: bool
    delay: float | None = None
    reason: str | None = None


# ---------------------------------------------------------------------------
# RetryPolicy type alias
# ---------------------------------------------------------------------------

# A policy is any callable that receives a RetryContext and returns either
# a plain bool (True = retry, False = give up) or a full RetryDecision.
# It may be sync or async.
RetryPolicy = Callable[[RetryContext], bool | RetryDecision | Awaitable[bool | RetryDecision]]


# ---------------------------------------------------------------------------
# Built-in policy factories  (mirrors OpenAI Agents SDK retry_policies object)
# ---------------------------------------------------------------------------


def never() -> RetryPolicy:
    """Never retry. Errors propagate immediately."""

    def policy(context: RetryContext) -> bool:
        return False

    return policy


def on_exception(*exception_types: type[BaseException]) -> RetryPolicy:
    """Retry only when the error is an instance of one of the given types."""

    def policy(context: RetryContext) -> bool:
        return isinstance(context.error, tuple(exception_types))

    return policy


def on_status_codes(*codes: int) -> RetryPolicy:
    """
    Retry when the exception carries an HTTP status code in ``codes``.
    Works with httpx, requests, and any exception that exposes
    ``status_code`` or ``response.status_code``.
    """

    def _extract_status(error: Exception) -> int | None:
        code = getattr(error, "status_code", None)
        if code is not None:
            return int(code)
        response = getattr(error, "response", None)
        if response is not None:
            code = getattr(response, "status_code", None)
            if code is not None:
                return int(code)
        return None

    allowed = frozenset(codes)

    def policy(context: RetryContext) -> bool:
        status = _extract_status(context.error)
        return status is not None and status in allowed

    return policy


def any_of(*policies: RetryPolicy) -> RetryPolicy:
    """Retry if **any** of the given policies says to retry."""

    async def policy(context: RetryContext) -> bool | RetryDecision:
        for p in policies:
            decision = await _evaluate(p, context)
            if decision.retry:
                return decision
        return RetryDecision(retry=False)

    return policy


def all_of(*policies: RetryPolicy) -> RetryPolicy:
    """Retry only if **all** of the given policies say to retry."""

    async def policy(context: RetryContext) -> bool | RetryDecision:
        last: RetryDecision = RetryDecision(retry=True)
        for p in policies:
            decision = await _evaluate(p, context)
            if not decision.retry:
                return decision
            last = decision
        return last

    return policy


# ---------------------------------------------------------------------------
# Backoff calculation
# ---------------------------------------------------------------------------


@dataclass
class BackoffSettings:
    """
    Controls the delay between retry attempts.

    The delay for attempt N is::

        min(initial_delay * multiplier^(N-1), max_delay)

    With jitter enabled the result is multiplied by a random factor in
    [0.875, 1.125] so concurrent callers spread their retries out.
    """

    initial_delay: float = DEFAULT_INITIAL_DELAY
    max_delay: float = DEFAULT_MAX_DELAY
    multiplier: float = DEFAULT_MULTIPLIER
    jitter: bool = DEFAULT_JITTER


def _compute_delay(attempt: int, settings: BackoffSettings) -> float:
    base = min(
        settings.initial_delay * (settings.multiplier ** max(attempt - 1, 0)),
        settings.max_delay,
    )
    if not settings.jitter:
        return base
    return min(max(base * (0.875 + random.random() * 0.25), 0.0), settings.max_delay)


# ---------------------------------------------------------------------------
# RetryStrategy — the object the loop holds
# ---------------------------------------------------------------------------


@dataclass
class RetryStrategy:
    """
    Combines a retry policy with backoff settings into one object the loop
    can call without knowing the details.

    Usage::

        strategy = RetryStrategy(
            max_attempts=6,
            policy=on_status_codes(429, 503),
        )

        result = await strategy.execute(lambda: model.chat(messages))

    ``execute`` wraps any async callable.  On failure it consults the policy,
    waits the computed delay, and tries again until the attempt ceiling is
    reached or the policy says to stop.
    """

    max_attempts: int = DEFAULT_MAX_ATTEMPTS
    policy: RetryPolicy = field(default_factory=never)
    backoff: BackoffSettings = field(default_factory=BackoffSettings)

    async def execute(
        self,
        fn: Callable[[], Awaitable[T]],
        **kwargs: Any,
    ) -> T:
        """
        Call ``fn`` and retry according to the policy.

        Args:
            fn: An async callable that takes no arguments.  Wrap your call
                in a lambda if you need to pass arguments::

                    await strategy.execute(lambda: model.chat(messages))

        Returns:
            The return value of ``fn`` on success.

        Raises:
            The last exception raised by ``fn`` when all attempts are
            exhausted or the policy says not to retry.
        """
        attempt = 1

        while True:
            try:
                return await fn()
            except Exception as error:
                context = RetryContext(
                    error=error,
                    attempt=attempt,
                    max_attempts=self.max_attempts,
                )

                if attempt >= self.max_attempts:
                    logger.debug(
                        "attempt=%d max_attempts=%d | attempts exhausted, not retrying",
                        attempt,
                        self.max_attempts,
                    )
                    raise

                decision = await _evaluate(self.policy, context)

                if not decision.retry:
                    logger.debug(
                        "attempt=%d error=%r reason=%s | policy said not to retry",
                        attempt,
                        error,
                        decision.reason or "none",
                    )
                    raise

                delay = (
                    decision.delay
                    if decision.delay is not None
                    else _compute_delay(attempt, self.backoff)
                )

                logger.debug(
                    "attempt=%d delay=%.3fs error=%r | retrying",
                    attempt,
                    delay,
                    error,
                )

                await asyncio.sleep(delay)
                attempt += 1


# ---------------------------------------------------------------------------
# Convenience constructors
# ---------------------------------------------------------------------------


def no_retry() -> RetryStrategy:
    """A RetryStrategy that never retries — errors propagate immediately."""
    return RetryStrategy(max_attempts=1, policy=never())


def exponential_backoff(
    *,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    policy: RetryPolicy | None = None,
    initial_delay: float = DEFAULT_INITIAL_DELAY,
    max_delay: float = DEFAULT_MAX_DELAY,
    multiplier: float = DEFAULT_MULTIPLIER,
    jitter: bool = DEFAULT_JITTER,
) -> RetryStrategy:
    """
    A RetryStrategy with exponential backoff and optional jitter.

    If ``policy`` is not provided it defaults to retrying on HTTP 429 and 503
    — the two most common transient errors in LLM APIs.
    """
    return RetryStrategy(
        max_attempts=max_attempts,
        policy=policy or on_status_codes(429, 503),
        backoff=BackoffSettings(
            initial_delay=initial_delay,
            max_delay=max_delay,
            multiplier=multiplier,
            jitter=jitter,
        ),
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _coerce(value: bool | RetryDecision) -> RetryDecision:
    if isinstance(value, RetryDecision):
        return value
    return RetryDecision(retry=bool(value))


async def _evaluate(policy: RetryPolicy, context: RetryContext) -> RetryDecision:
    raw = policy(context)
    if asyncio.iscoroutine(raw):
        resolved: bool | RetryDecision = await cast("Awaitable[bool | RetryDecision]", raw)
    else:
        resolved = cast("bool | RetryDecision", raw)
    return _coerce(resolved)

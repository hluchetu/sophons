from __future__ import annotations

import json
import math
from typing import Any

from sophons.errors import (
    ContextWindowOverflowError,
    ModelAuthenticationError,
    ModelAuthorizationError,
    ModelError,
    ModelInvalidRequestError,
    ModelResponseError,
    ModelThrottledError,
    ModelTimeoutError,
    ModelUnavailableError,
    is_context_overflow,
)
from sophons.integrations.models.adapters.openai_compat import OpenAICompatAdapter
from collections.abc import Iterator

from sophons.models.messages import Message
from sophons.models.streaming import (
    MessageStop,
    ReasoningDelta,
    StreamEvent,
    TextDelta,
    ToolCallDelta,
    UsageUpdate,
)
from sophons.tools.base import Tool

# OpenAI-format finish reasons to Sophons stop reasons.
_STOP_REASONS = {
    "stop": "end_turn",
    "tool_calls": "tool_use",
    "length": "max_tokens",
    "content_filter": "content_filtered",
}


class DeepSeekModel:
    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str = "https://api.deepseek.com/v1",
        thinking: bool = False,
        context_window: int | None = None,
        *,
        timeout_seconds: float = 60.0,
        sdk_max_retries: int = 0,
    ) -> None:
        if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float)) or not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be finite and positive")
        if type(sdk_max_retries) is not int or sdk_max_retries < 0:
            raise ValueError("sdk_max_retries must be a nonnegative integer")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "DeepSeekModel requires the 'deepseek' extra. "
                "Install it with: uv sync --extra deepseek"
            ) from exc

        self.model = model
        # Total context size in tokens, read by ratio-based conversation
        # managers. Left unset by default rather than guessed: the figure
        # differs per model and changes between releases, and a wrong value
        # is worse than an absent one — strategies fall back to absolute
        # thresholds when this is None.
        self.context_window = context_window
        # Let the agent RetryStrategy own retries by default. Hidden provider
        # retries otherwise multiply requests behind one model.invoke attempt.
        self._client = OpenAI(api_key=api_key, base_url=base_url,
                              timeout=timeout_seconds, max_retries=sdk_max_retries)
        self._thinking = thinking
        self._adapter = OpenAICompatAdapter()

    def invoke(
        self, messages: list[Message], tools: list[Tool] | None = None
    ) -> Message:
        kwargs: dict = dict(
            model=self.model,
            messages=self._adapter.serialize_messages(messages),
            temperature=0,
            stream=False,
        )
        if self._thinking:
            kwargs["extra_body"] = {"thinking": {"type": "enabled"}}
        if tools:
            kwargs["tools"] = self._adapter.serialize_tools(tools)

        try:
            response = self._client.chat.completions.create(**kwargs)
        except Exception as exc:
            raise _translate_error(exc, model=self.model) from exc

        message = _extract_message(response, model=self.model)

        tool_calls = _normalize_tool_calls(getattr(message, "tool_calls", None) or [])
        metadata = {"tool_calls": tool_calls} if tool_calls else {}

        if getattr(message, "reasoning_content", None):
            metadata["reasoning"] = message.reasoning_content

        if response.usage is not None:
            usage = {
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
            }
            cache_hit = getattr(response.usage, "prompt_cache_hit_tokens", None)
            if cache_hit is not None:
                usage["cache_read_tokens"] = cache_hit
            metadata["usage"] = usage

        return Message(
            role="assistant", content=message.content or "", metadata=metadata
        )

    def stream(
        self, messages: list[Message], tools: list[Tool] | None = None
    ) -> Iterator[StreamEvent]:
        """The same request as ``invoke``, delivered as stream events.

        A stream that goes silent is ended by the client's read timeout and
        raised as ``ModelTimeoutError``. Closing the iterator closes the request.
        """
        kwargs: dict = dict(
            model=self.model,
            messages=self._adapter.serialize_messages(messages),
            temperature=0,
            stream=True,
            stream_options={"include_usage": True},
        )
        if self._thinking:
            kwargs["extra_body"] = {"thinking": {"type": "enabled"}}
        if tools:
            kwargs["tools"] = self._adapter.serialize_tools(tools)
        try:
            chunks = self._client.chat.completions.create(**kwargs)
        except Exception as exc:
            raise _translate_error(exc, model=self.model) from exc
        try:
            try:
                yield from _stream_events(chunks, model=self.model)
            except ModelError:
                raise
            except Exception as exc:
                raise _translate_error(exc, model=self.model) from exc
        finally:
            close = getattr(chunks, "close", None)
            if close is not None:
                close()


def _stream_events(chunks: Any, *, model: str) -> Iterator[StreamEvent]:
    """Translate OpenAI-format stream chunks into Sophons stream events."""

    stopped = False
    for chunk in chunks:
        usage = getattr(chunk, "usage", None)
        if usage is not None:
            yield UsageUpdate(
                input_tokens=getattr(usage, "prompt_tokens", None),
                output_tokens=getattr(usage, "completion_tokens", None),
                cache_read_tokens=getattr(usage, "prompt_cache_hit_tokens", None),
            )
        choices = getattr(chunk, "choices", None) or []
        if not choices:
            continue  # the usage-only chunk that closes the stream
        choice = choices[0]
        delta = getattr(choice, "delta", None)
        if delta is not None:
            reasoning = getattr(delta, "reasoning_content", None)
            if reasoning:
                yield ReasoningDelta(reasoning)
            content = getattr(delta, "content", None)
            if content:
                yield TextDelta(content)
            for position, call in enumerate(getattr(delta, "tool_calls", None) or []):
                function = getattr(call, "function", None)
                index = getattr(call, "index", None)
                yield ToolCallDelta(
                    index=index if isinstance(index, int) else position,
                    tool_use_id=getattr(call, "id", None) or None,
                    name=getattr(function, "name", None) or None,
                    arguments=getattr(function, "arguments", None) or "",
                )
        finish = getattr(choice, "finish_reason", None)
        if finish:
            stopped = True
            yield MessageStop(_STOP_REASONS.get(finish, finish))
    if not stopped:
        raise ModelResponseError(
            "DeepSeek ended the stream without a finish reason.",
            details={"provider": "deepseek", "model": model},
        )


def _translate_error(error: Exception, *, model: str) -> ModelError:
    error_name = type(error).__name__
    status_code = getattr(error, "status_code", None)
    request_id = getattr(error, "request_id", None)

    details: dict[str, Any] = {
        "provider": "deepseek",
        "model": model,
    }
    if status_code is not None:
        details["status_code"] = status_code
    if request_id is not None:
        details["request_id"] = request_id

    if is_context_overflow(error):
        return ContextWindowOverflowError(
            "The DeepSeek request exceeded the model context window.",
            details=details,
        )
    if error_name == "AuthenticationError" or status_code == 401:
        return ModelAuthenticationError(
            "DeepSeek authentication failed.",
            details=details,
        )
    if error_name == "PermissionDeniedError" or status_code == 403:
        return ModelAuthorizationError(
            "DeepSeek rejected the request because permission was denied.",
            details=details,
        )
    if error_name == "RateLimitError" or status_code == 429:
        return ModelThrottledError(
            "DeepSeek rate-limited the request.",
            details=details,
        )
    if error_name == "APITimeoutError":
        return ModelTimeoutError(
            "The DeepSeek request timed out.",
            details=details,
        )
    if error_name == "APIConnectionError":
        return ModelUnavailableError(
            "Could not connect to DeepSeek.",
            details=details,
        )
    if error_name == "BadRequestError" or status_code == 400:
        return ModelInvalidRequestError(
            "DeepSeek rejected the request as invalid.",
            details=details,
        )
    if error_name == "InternalServerError" or (
        isinstance(status_code, int) and status_code >= 500
    ):
        return ModelUnavailableError(
            "DeepSeek is temporarily unavailable.",
            details=details,
        )
    return ModelError(
        "The DeepSeek request failed.",
        details={**details, "provider_error_type": error_name},
    )


def _extract_message(response: object, *, model: str) -> Any:
    choices = getattr(response, "choices", None)
    if not choices:
        raise ModelResponseError(
            "DeepSeek returned a response without any choices.",
            details={"provider": "deepseek", "model": model},
        )

    message = getattr(choices[0], "message", None)
    if message is None:
        raise ModelResponseError(
            "DeepSeek returned a choice without a message.",
            details={"provider": "deepseek", "model": model},
        )
    return message


def _normalize_tool_calls(raw: list) -> list[dict]:
    result = []
    for item in raw:
        fn = getattr(item, "function", None)
        name = getattr(fn, "name", None)
        if not name:
            continue
        args = getattr(fn, "arguments", {}) or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {"raw": args}
        result.append(
            {"tool_use_id": getattr(item, "id", None), "name": name, "input": args}
        )
    return result

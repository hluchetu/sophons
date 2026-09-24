from __future__ import annotations

from types import SimpleNamespace

import pytest

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
)
from sophons.integrations.models.deepseek import (
    _extract_message,
    _translate_error,
)


@pytest.mark.parametrize(
    ("error_name", "status_code", "expected_type"),
    [
        ("AuthenticationError", 401, ModelAuthenticationError),
        ("PermissionDeniedError", 403, ModelAuthorizationError),
        ("RateLimitError", 429, ModelThrottledError),
        ("APITimeoutError", None, ModelTimeoutError),
        ("APIConnectionError", None, ModelUnavailableError),
        ("BadRequestError", 400, ModelInvalidRequestError),
        ("InternalServerError", 500, ModelUnavailableError),
    ],
)
def test_translates_known_provider_errors(
    error_name: str,
    status_code: int | None,
    expected_type: type[ModelError],
) -> None:
    error_type = type(error_name, (Exception,), {})
    provider_error = error_type("provider failure")
    provider_error.status_code = status_code
    provider_error.request_id = "request-123"

    translated = _translate_error(provider_error, model="deepseek-chat")

    assert isinstance(translated, expected_type)
    expected_details = {
        "provider": "deepseek",
        "model": "deepseek-chat",
        "request_id": "request-123",
    }
    if status_code is not None:
        expected_details["status_code"] = status_code
    assert translated.details == expected_details


def test_translates_context_overflow_before_bad_request() -> None:
    error_type = type("BadRequestError", (Exception,), {})
    provider_error = error_type("maximum context length exceeded")
    provider_error.status_code = 400

    translated = _translate_error(provider_error, model="deepseek-chat")

    assert isinstance(translated, ContextWindowOverflowError)


def test_unknown_provider_error_uses_model_error_fallback() -> None:
    class UnexpectedProviderError(Exception):
        pass

    translated = _translate_error(
        UnexpectedProviderError("provider internals"),
        model="deepseek-chat",
    )

    assert type(translated) is ModelError
    assert str(translated) == "The DeepSeek request failed."
    assert translated.details["provider_error_type"] == "UnexpectedProviderError"


def test_extract_message_rejects_missing_choices() -> None:
    with pytest.raises(ModelResponseError, match="without any choices"):
        _extract_message(SimpleNamespace(choices=[]), model="deepseek-chat")


def test_extract_message_rejects_missing_message() -> None:
    response = SimpleNamespace(choices=[SimpleNamespace(message=None)])

    with pytest.raises(ModelResponseError, match="without a message"):
        _extract_message(response, model="deepseek-chat")


def test_extract_message_returns_provider_message() -> None:
    message = SimpleNamespace(content="hello")
    response = SimpleNamespace(choices=[SimpleNamespace(message=message)])

    assert _extract_message(response, model="deepseek-chat") is message

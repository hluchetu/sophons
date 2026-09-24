import pytest

from sophons.errors import (
    ContextWindowOverflowError,
    ErrorCode,
    ModelAuthenticationError,
    ModelError,
    ModelThrottledError,
    ModelTimeoutError,
    is_context_overflow,
)


@pytest.mark.parametrize(
    ("error_type", "expected_code"),
    [
        (
            ModelAuthenticationError,
            ErrorCode.MODEL_AUTHENTICATION_ERROR,
        ),
        (
            ModelThrottledError,
            ErrorCode.MODEL_THROTTLED_ERROR,
        ),
        (
            ModelTimeoutError,
            ErrorCode.MODEL_TIMEOUT_ERROR,
        ),
    ],
)
def test_model_errors_have_stable_codes(
    error_type,
    expected_code,
) -> None:
    error = error_type(
        "provider failed",
        details={"provider": "deepseek"},
    )

    assert isinstance(error, ModelError)
    assert error.error_code is expected_code
    assert error.details == {"provider": "deepseek"}
    assert str(error) == "provider failed"


def test_context_overflow_is_a_model_error() -> None:
    error = ContextWindowOverflowError("context is too large")

    assert isinstance(error, ModelError)
    assert is_context_overflow(error)

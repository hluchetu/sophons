from __future__ import annotations

from enum import Enum


class ErrorCode(str, Enum):
    """Stable Sophons error codes for docs, logs, and integrations."""

    CONFIGURATION_ERROR = "CONFIGURATION_ERROR"
    CONTEXT_WINDOW_OVERFLOW = "CONTEXT_WINDOW_OVERFLOW"
    INTEGRATION_ERROR = "INTEGRATION_ERROR"
    LOADER_ERROR = "LOADER_ERROR"
    MISSING_DEPENDENCY = "MISSING_DEPENDENCY"
    RETRIEVER_ERROR = "RETRIEVER_ERROR"
    SPLITTER_ERROR = "SPLITTER_ERROR"
    TOOL_ERROR = "TOOL_ERROR"
    UNSUPPORTED_FILE_TYPE = "UNSUPPORTED_FILE_TYPE"


class SophonsError(Exception):
    """Base class for Sophons exceptions."""

    error_code: ErrorCode

    def __init__(
        self,
        message: str,
        *,
        error_code: ErrorCode,
        details: dict | None = None,
    ) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.details = dict(details or {})


class LoaderError(SophonsError):
    """Base class for loader errors."""

    def __init__(
        self,
        message: str,
        *,
        error_code: ErrorCode = ErrorCode.LOADER_ERROR,
        details: dict | None = None,
    ) -> None:
        super().__init__(message, error_code=error_code, details=details)


class ConfigurationError(SophonsError):
    """Raised when a Sophons component is configured incorrectly."""

    def __init__(self, message: str, *, details: dict | None = None) -> None:
        super().__init__(
            message,
            error_code=ErrorCode.CONFIGURATION_ERROR,
            details=details,
        )


class MissingDependencyError(SophonsError):
    """Raised when an optional integration dependency is missing."""

    def __init__(self, message: str, *, details: dict | None = None) -> None:
        super().__init__(
            message,
            error_code=ErrorCode.MISSING_DEPENDENCY,
            details=details,
        )


class UnsupportedFileTypeError(LoaderError):
    """Raised when Sophons does not have a loader for a file type."""

    def __init__(self, message: str, *, details: dict | None = None) -> None:
        super().__init__(
            message,
            error_code=ErrorCode.UNSUPPORTED_FILE_TYPE,
            details=details,
        )


class SplitterError(SophonsError):
    """Base class for splitter errors."""

    def __init__(self, message: str, *, details: dict | None = None) -> None:
        super().__init__(
            message,
            error_code=ErrorCode.SPLITTER_ERROR,
            details=details,
        )


class RetrieverError(SophonsError):
    """Base class for retriever errors."""

    def __init__(self, message: str, *, details: dict | None = None) -> None:
        super().__init__(
            message,
            error_code=ErrorCode.RETRIEVER_ERROR,
            details=details,
        )


class ToolError(SophonsError):
    """Base class for tool errors."""

    def __init__(self, message: str, *, details: dict | None = None) -> None:
        super().__init__(message, error_code=ErrorCode.TOOL_ERROR, details=details)


class ContextWindowOverflowError(SophonsError):
    """
    Raised when the model rejects a request for exceeding its context window.

    This is recoverable, unlike most provider errors: the agent asks its
    conversation manager to reduce the context and tries again. Provider
    adapters map their own "prompt is too long" errors onto this so the loop
    does not have to know each provider's wording.
    """

    def __init__(self, message: str, *, details: dict | None = None) -> None:
        super().__init__(
            message,
            error_code=ErrorCode.CONTEXT_WINDOW_OVERFLOW,
            details=details,
        )


_OVERFLOW_MARKERS = (
    "context length",
    "context_length_exceeded",
    "context window",
    "maximum context",
    "prompt is too long",
    "too many tokens",
    "reduce the length",
    "string too long",
)


def is_context_overflow(error: Exception) -> bool:
    """
    Guess whether a provider error means "your request was too large".

    Providers report this inconsistently and none of them use a shared error
    code, so matching on message text is the available option. False
    negatives merely mean the run fails as it does today; false positives
    cost one wasted retry with a smaller context.
    """
    if isinstance(error, ContextWindowOverflowError):
        return True
    text = str(error).lower()
    return any(marker in text for marker in _OVERFLOW_MARKERS)


class IntegrationError(SophonsError):
    """Base class for integration errors."""

    def __init__(self, message: str, *, details: dict | None = None) -> None:
        super().__init__(
            message,
            error_code=ErrorCode.INTEGRATION_ERROR,
            details=details,
        )

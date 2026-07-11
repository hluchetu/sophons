from __future__ import annotations

from enum import Enum


class ErrorCode(str, Enum):
    """Stable Sophons error codes for docs, logs, and integrations."""

    CONFIGURATION_ERROR = "CONFIGURATION_ERROR"
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


class IntegrationError(SophonsError):
    """Base class for integration errors."""

    def __init__(self, message: str, *, details: dict | None = None) -> None:
        super().__init__(
            message,
            error_code=ErrorCode.INTEGRATION_ERROR,
            details=details,
        )

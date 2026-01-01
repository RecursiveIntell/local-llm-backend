"""Custom exception hierarchy for Aphelion Magnetar.

This module defines a structured exception hierarchy that provides:
- Clear error categorization
- Consistent error handling patterns
- Rich error context for debugging
"""

from __future__ import annotations

from typing import Any


class AphelionError(Exception):
    """Base exception for all Aphelion errors.

    All custom exceptions in the application inherit from this class,
    allowing for catch-all exception handling when needed.

    Attributes:
        message: Human-readable error description.
        details: Optional dictionary with additional context.
    """

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        """Initialize the exception.

        Args:
            message: Human-readable error description.
            details: Optional dictionary with additional context.
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        """Return string representation with details if present."""
        if self.details:
            detail_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} ({detail_str})"
        return self.message


class ConfigurationError(AphelionError):
    """Raised when there's a configuration-related error.

    Examples:
        - Missing required configuration
        - Invalid configuration values
        - Failed to load/save configuration
    """


class ProviderError(AphelionError):
    """Base exception for LLM provider-related errors.

    Attributes:
        provider_name: Name of the provider that raised the error.
    """

    def __init__(
        self,
        message: str,
        provider_name: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the provider error.

        Args:
            message: Human-readable error description.
            provider_name: Name of the provider that raised the error.
            details: Optional dictionary with additional context.
        """
        details = details or {}
        details["provider"] = provider_name
        super().__init__(message, details)
        self.provider_name = provider_name


class ProviderConnectionError(ProviderError):
    """Raised when a provider connection fails.

    Examples:
        - Network unreachable
        - Invalid API endpoint
        - Authentication failure
    """


# Alias for backward compatibility
ConnectionError = ProviderConnectionError


class RateLimitError(ProviderError):
    """Raised when a provider rate limit is hit.

    Attributes:
        retry_after: Seconds to wait before retrying (if provided by API).
    """

    def __init__(
        self,
        message: str,
        provider_name: str,
        retry_after: float | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the rate limit error.

        Args:
            message: Human-readable error description.
            provider_name: Name of the provider.
            retry_after: Seconds to wait before retrying.
            details: Optional dictionary with additional context.
        """
        details = details or {}
        if retry_after is not None:
            details["retry_after"] = retry_after
        super().__init__(message, provider_name, details)
        self.retry_after = retry_after


class ModelNotFoundError(ProviderError):
    """Raised when a requested model is not available."""


class InvalidAPIKeyError(ProviderError):
    """Raised when an API key is invalid or expired."""


class FileProcessingError(AphelionError):
    """Raised when file processing fails.

    Examples:
        - Unsupported file format
        - File too large
        - Corrupted file
    """

    def __init__(
        self,
        message: str,
        file_path: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the file processing error.

        Args:
            message: Human-readable error description.
            file_path: Path to the file that caused the error.
            details: Optional dictionary with additional context.
        """
        details = details or {}
        if file_path:
            details["file_path"] = file_path
        super().__init__(message, details)
        self.file_path = file_path


class ConversationError(AphelionError):
    """Raised when conversation operations fail.

    Examples:
        - Conversation not found
        - Failed to save conversation
        - Context window exceeded
    """


class ServerError(AphelionError):
    """Raised when the remote access server encounters an error."""


class AuthenticationError(AphelionError):
    """Raised when authentication fails."""

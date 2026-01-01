"""Core module for Aphelion Magnetar."""

from dodger.core.exceptions import (
    AphelionError,
    ConfigurationError,
    ConnectionError,
    ProviderError,
    RateLimitError,
)

__all__ = [
    "AphelionError",
    "ConfigurationError",
    "ConnectionError",
    "ProviderError",
    "RateLimitError",
]

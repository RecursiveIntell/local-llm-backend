"""Aphelion Magnetar - Multi-Provider AI Chat Application."""

from dodger.core.exceptions import (
    AphelionError,
    ConfigurationError,
    ConnectionError,
    ProviderError,
    RateLimitError,
)

__version__ = "0.1.0"
__all__ = [
    "AphelionError",
    "ConfigurationError",
    "ConnectionError",
    "ProviderError",
    "RateLimitError",
]

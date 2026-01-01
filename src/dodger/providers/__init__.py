"""LLM Provider implementations for Aphelion Magnetar."""

from dodger.providers.anthropic_provider import AnthropicProvider
from dodger.providers.base import BaseProvider, ProviderCapabilities
from dodger.providers.google_provider import GoogleProvider
from dodger.providers.lmstudio_provider import LMStudioProvider
from dodger.providers.manager import ProviderManager
from dodger.providers.ollama_provider import OllamaProvider
from dodger.providers.openai_provider import OpenAIProvider
from dodger.providers.openrouter_provider import OpenRouterProvider

__all__ = [
    "AnthropicProvider",
    "BaseProvider",
    "GoogleProvider",
    "LMStudioProvider",
    "OllamaProvider",
    "OpenAIProvider",
    "OpenRouterProvider",
    "ProviderCapabilities",
    "ProviderManager",
]

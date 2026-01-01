"""Provider manager for unified access to all LLM providers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog

from dodger.config import ProviderType, config_manager
from dodger.core.exceptions import ProviderError
from dodger.core.telemetry import telemetry
from dodger.providers.anthropic_provider import AnthropicProvider
from dodger.providers.google_provider import GoogleProvider
from dodger.providers.lmstudio_provider import LMStudioProvider
from dodger.providers.ollama_provider import OllamaProvider
from dodger.providers.openai_provider import OpenAIProvider
from dodger.providers.openrouter_provider import OpenRouterProvider

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from dodger.providers.base import (
        BaseProvider,
        ChatMessage,
        ChatResponse,
        ModelInfo,
        StreamChunk,
    )

logger = structlog.get_logger(__name__)


# Provider type to class mapping
PROVIDER_CLASSES: dict[ProviderType, type[BaseProvider]] = {
    ProviderType.OPENAI: OpenAIProvider,
    ProviderType.ANTHROPIC: AnthropicProvider,
    ProviderType.GOOGLE: GoogleProvider,
    ProviderType.OPENROUTER: OpenRouterProvider,
    ProviderType.OLLAMA: OllamaProvider,
    ProviderType.LMSTUDIO: LMStudioProvider,
}


class ProviderManager:
    """Manages all LLM providers with unified access.

    This class provides:
    - Provider initialization and caching
    - Unified chat interface across providers
    - Automatic fallback between providers
    - Model discovery aggregation
    """

    def __init__(self) -> None:
        """Initialize the provider manager."""
        self._providers: dict[ProviderType, BaseProvider] = {}
        self._provider_configs: dict[ProviderType, tuple[str | None, str | None]] = {}
        self._active_provider: ProviderType | None = None
        self._active_model: str | None = None

    def _get_provider(self, provider_type: ProviderType) -> BaseProvider:
        """Get or create a provider instance.

        Args:
            provider_type: The type of provider to get.

        Returns:
            The provider instance.
        """
        settings = config_manager.settings
        provider_config = settings.providers[provider_type]

        # Check if config changed (base_url or api_key)
        current_config = (
            provider_config.base_url,
            provider_config.api_key.get_secret_value() if provider_config.api_key else None,
        )
        cached_config = self._provider_configs.get(provider_type)

        if provider_type in self._providers and cached_config == current_config:
            return self._providers[provider_type]

        # Config changed or not cached, create new provider
        provider_class = PROVIDER_CLASSES[provider_type]

        api_key = None
        if provider_config.api_key:
            api_key = provider_config.api_key.get_secret_value()

        logger.info(
            "Creating provider instance",
            provider=provider_type.value,
            base_url=provider_config.base_url,
        )

        self._providers[provider_type] = provider_class(
            api_key=api_key,
            base_url=provider_config.base_url,
            default_model=provider_config.default_model,
        )
        self._provider_configs[provider_type] = current_config

        return self._providers[provider_type]

    def invalidate_provider(self, provider_type: ProviderType) -> None:
        """Invalidate a cached provider, forcing recreation on next use.

        Args:
            provider_type: The provider to invalidate.
        """
        if provider_type in self._providers:
            del self._providers[provider_type]
        if provider_type in self._provider_configs:
            del self._provider_configs[provider_type]
        logger.info("Invalidated provider cache", provider=provider_type.value)

    async def refresh_models(self, provider_type: ProviderType) -> list[ModelInfo]:
        """Refresh available models for a provider.

        Args:
            provider_type: The provider to refresh.

        Returns:
            List of available models.
        """
        # Invalidate cache to pick up any config changes
        self.invalidate_provider(provider_type)

        try:
            provider = self._get_provider(provider_type)
            models = await provider.get_available_models()

            # Update config with available models
            model_ids = [m.id for m in models]
            config_manager.update_provider(
                provider_type,
                available_models=model_ids,
            )

            logger.info(
                "Refreshed models",
                provider=provider_type.value,
                count=len(models),
            )
            return models
        except Exception as e:
            logger.exception("Failed to refresh models", provider=provider_type.value)
            return []

    def get_active_provider(self) -> BaseProvider:
        """Get the currently active provider.

        Returns:
            The active provider instance.

        Raises:
            ProviderError: If no provider is configured.
        """
        settings = config_manager.settings
        provider_type = self._active_provider or settings.active_provider

        if not settings.providers[provider_type].enabled:
            # Find first enabled provider
            for pt, config in settings.providers.items():
                if config.enabled:
                    provider_type = pt
                    break
            else:
                msg = "No providers configured. Please add an API key in Settings."
                raise ProviderError(
                    msg,
                    provider_name="none",
                )

        return self._get_provider(provider_type)

    def set_active_provider(self, provider_type: ProviderType) -> None:
        """Set the active provider.

        Args:
            provider_type: The provider type to activate.
        """
        self._active_provider = provider_type
        settings = config_manager.settings
        settings.active_provider = provider_type
        config_manager.save()
        logger.info("Switched provider", provider=provider_type.value)

    def set_active_model(self, model: str) -> None:
        """Set the active model.

        Args:
            model: The model ID to use.
        """
        self._active_model = model
        settings = config_manager.settings
        settings.active_model = model
        config_manager.save()
        logger.info("Switched model", model=model)

    async def get_all_available_models(self) -> dict[ProviderType, list[ModelInfo]]:
        """Get available models from all enabled providers.

        Returns:
            Dictionary mapping provider types to their available models.
        """
        settings = config_manager.settings
        result: dict[ProviderType, list[ModelInfo]] = {}

        for provider_type, config in settings.providers.items():
            if config.enabled:
                try:
                    provider = self._get_provider(provider_type)
                    models = await provider.get_available_models()
                    result[provider_type] = models
                except Exception as e:
                    logger.warning(
                        "Failed to get models",
                        provider=provider_type.value,
                        error=str(e),
                    )
                    result[provider_type] = []

        return result

    async def validate_provider(self, provider_type: ProviderType) -> bool:
        """Validate a provider's connection.

        Args:
            provider_type: The provider to validate.

        Returns:
            True if the provider is working.
        """
        try:
            provider = self._get_provider(provider_type)
            return await provider.validate_connection()
        except Exception as e:
            logger.warning(
                "Provider validation failed",
                provider=provider_type.value,
                error=str(e),
            )
            return False

    def _get_provider_for_model(self, model_id: str) -> BaseProvider:
        """Find the provider that owns the given model.

        Args:
            model_id: The model ID to look up.

        Returns:
            The provider for the model, or the active provider if not found.
        """
        settings = config_manager.settings

        # Check all enabled providers
        for provider_type, config in settings.providers.items():
            if config.enabled and model_id in config.available_models:
                return self._get_provider(provider_type)

        # Fallback to active provider if not found
        logger.warning("Model not found in any provider, using active", model=model_id)
        return self.get_active_provider()

    async def chat(
        self,
        messages: list[ChatMessage],
        model: str | None = None,
        provider: ProviderType | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """Send a chat completion request.

        Args:
            messages: List of chat messages.
            model: Model to use (defaults to active model).
            provider: Provider to use (defaults to active provider).
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            **kwargs: Additional provider-specific options.

        Returns:
            The chat response.
        """
        # Determine target provider
        if provider:
            target_provider = self._get_provider(provider)
        elif model:
            target_provider = self._get_provider_for_model(model)
        else:
            target_provider = self.get_active_provider()

        model = model or self._active_model

        # Determine provider name for telemetry
        provider_name = None
        if provider:
            provider_name = provider.value
        elif hasattr(target_provider, "PROVIDER_TYPE"):
            # Best effort to get provider name if we resolved it from model
            # This relies on the provider instance having a way to identify itself,
            # which they don't explicitly have in the base class, but we can verify.
            # actually, let's just use the active provider as fallback for name logging
            provider_name = self._active_provider
        
        # Determine provider name by checking which provider instance we have
        # This is a bit inefficient but accurate
        if not provider_name:
             for p_type, p_inst in self._providers.items():
                 if p_inst is target_provider:
                     provider_name = p_type.value
                     break
        
        if not provider_name and self._active_provider:
             provider_name = self._active_provider.value

        async with telemetry.trace_async(
            "chat_completion",
            model=model,
            provider=provider_name,
            metadata={"message_count": len(messages)},
        ) as trace:
            response = await target_provider.chat(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )

            # Record usage
            telemetry.record_usage(
                trace,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
            )

            return response

    async def stream_chat(
        self,
        messages: list[ChatMessage],
        model: str | None = None,
        provider: ProviderType | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[StreamChunk, None]:
        """Stream a chat completion.

        Args:
            messages: List of chat messages.
            model: Model to use (defaults to active model).
            provider: Provider to use (defaults to active provider).
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            **kwargs: Additional provider-specific options.

        Yields:
            Stream chunks as they're generated.
        """
        # Determine target provider
        if provider:
            target_provider = self._get_provider(provider)
        elif model:
            target_provider = self._get_provider_for_model(model)
        else:
            target_provider = self.get_active_provider()

        model = model or self._active_model
        
        # Determine provider name for telemetry
        provider_name = None
        if provider:
            provider_name = provider.value
        
        if not provider_name:
             for p_type, p_inst in self._providers.items():
                 if p_inst is target_provider:
                     provider_name = p_type.value
                     break
        
        if not provider_name and self._active_provider:
             provider_name = self._active_provider.value

        async with telemetry.trace_async(
            "chat_stream",
            model=model,
            provider=provider_name,
            metadata={"message_count": len(messages)},
        ) as trace:
            total_input = 0
            total_output = 0

            async for chunk in target_provider.stream_chat(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            ):
                if chunk.is_final:
                    total_input = chunk.input_tokens
                    total_output = chunk.output_tokens
                yield chunk

            telemetry.record_usage(
                trace,
                input_tokens=total_input,
                output_tokens=total_output,
            )

    async def auto_detect_local_providers(self) -> dict[ProviderType, bool]:
        """Auto-detect and configure local LLM providers.

        Returns:
            Dictionary of provider types to detection status.
        """
        return await config_manager.auto_detect_local_llms()


# Global provider manager instance
provider_manager = ProviderManager()

"""OpenRouter provider implementation.

OpenRouter provides unified access to multiple LLM providers
through a single API compatible with OpenAI's format.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import httpx
import structlog

from dodger.core.exceptions import ConnectionError, InvalidAPIKeyError, ProviderError
from dodger.providers.base import (
    BaseProvider,
    ChatMessage,
    ChatResponse,
    ModelInfo,
    ProviderCapabilities,
    StreamChunk,
)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

logger = structlog.get_logger(__name__)


class OpenRouterProvider(BaseProvider):
    """OpenRouter API provider.

    Provides access to multiple LLM providers (OpenAI, Anthropic,
    Google, Meta, Mistral, etc.) through a unified API.
    """

    name = "openrouter"
    display_name = "OpenRouter"

    MODELS_ENDPOINT = "https://openrouter.ai/api/v1/models"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        default_model: str | None = "anthropic/claude-3.5-sonnet",
        **kwargs: Any,
    ) -> None:
        """Initialize the OpenRouter provider."""
        super().__init__(
            api_key=api_key,
            base_url=base_url or "https://openrouter.ai/api/v1",
            default_model=default_model,
            **kwargs,
        )
        self._models_cache: list[ModelInfo] | None = None

    @property
    def capabilities(self) -> ProviderCapabilities:
        """Get OpenRouter capabilities."""
        return ProviderCapabilities(
            streaming=True,
            vision=True,  # Depends on model
            function_calling=True,  # Depends on model
            json_mode=True,
            system_messages=True,
            max_context_window=200000,  # Varies by model
        )

    async def validate_connection(self) -> bool:
        """Validate the OpenRouter connection."""
        if not self.api_key:
            msg = "OpenRouter API key is required"
            raise InvalidAPIKeyError(
                msg,
                provider_name=self.name,
            )

        try:
            # Test by fetching models (cheaper than a completion)
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    self.MODELS_ENDPOINT,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    timeout=10.0,
                )
                if response.status_code == 401:
                    msg = "Invalid OpenRouter API key"
                    raise InvalidAPIKeyError(
                        msg,
                        provider_name=self.name,
                    )
                response.raise_for_status()
                return True

        except httpx.HTTPStatusError as e:
            msg = f"OpenRouter API error: {e.response.status_code}"
            raise ConnectionError(
                msg,
                provider_name=self.name,
            ) from e
        except Exception as e:
            if isinstance(e, InvalidAPIKeyError):
                raise
            msg = f"Failed to connect to OpenRouter: {e}"
            raise ConnectionError(
                msg,
                provider_name=self.name,
            ) from e

    async def get_available_models(self) -> list[ModelInfo]:
        """Get available models from OpenRouter API."""
        if self._models_cache is not None:
            return self._models_cache

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    self.MODELS_ENDPOINT,
                    headers={"Authorization": f"Bearer {self.api_key}"}
                    if self.api_key
                    else {},
                    timeout=10.0,
                )
                response.raise_for_status()
                data = response.json()

            models: list[ModelInfo] = []
            for model_data in data.get("data", []):
                model_id = model_data.get("id", "")
                pricing = model_data.get("pricing", {})

                models.append(
                    ModelInfo(
                        id=model_id,
                        name=model_data.get("name", model_id),
                        context_window=model_data.get("context_length", 4096),
                        supports_vision="vision" in model_data.get("architecture", {}).get(
                            "modality", ""
                        ),
                        supports_function_calling=model_data.get(
                            "supports_tool_use", False
                        ),
                        input_cost_per_1m=float(pricing.get("prompt", 0)) * 1_000_000,
                        output_cost_per_1m=float(pricing.get("completion", 0))
                        * 1_000_000,
                    )
                )

            # Sort by popularity (approximated by name familiarity)
            self._models_cache = sorted(models, key=lambda m: m.name)
            return self._models_cache

        except Exception as e:
            logger.warning("Failed to fetch OpenRouter models", error=str(e))
            # Return some common models as fallback
            return [
                ModelInfo(
                    id="anthropic/claude-3.5-sonnet",
                    name="Claude 3.5 Sonnet",
                    context_window=200000,
                    supports_vision=True,
                ),
                ModelInfo(
                    id="openai/gpt-4o",
                    name="GPT-4o",
                    context_window=128000,
                    supports_vision=True,
                ),
                ModelInfo(
                    id="google/gemini-pro-1.5",
                    name="Gemini 1.5 Pro",
                    context_window=1000000,
                    supports_vision=True,
                ),
            ]

    async def chat(
        self,
        messages: list[ChatMessage],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """Send a chat completion request via OpenRouter."""
        model = self._get_model(model)
        lc_messages = self._convert_messages(messages)

        try:
            from langchain_openai import ChatOpenAI

            # OpenRouter uses OpenAI-compatible API
            client = ChatOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                default_headers={
                    "HTTP-Referer": "https://github.com/aphelion-magnetar",
                    "X-Title": "Aphelion Magnetar",
                },
            )

            response = await client.ainvoke(lc_messages, **kwargs)

            input_tokens = 0
            output_tokens = 0
            if hasattr(response, "response_metadata"):
                usage = response.response_metadata.get("token_usage", {})
                input_tokens = usage.get("prompt_tokens", 0)
                output_tokens = usage.get("completion_tokens", 0)

            return ChatResponse(
                content=str(response.content),
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )

        except Exception as e:
            logger.exception("OpenRouter chat error", error=str(e), model=model)
            msg = f"OpenRouter chat failed: {e}"
            raise ProviderError(
                msg,
                provider_name=self.name,
            ) from e

    async def stream_chat(
        self,
        messages: list[ChatMessage],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[StreamChunk, None]:
        """Stream a chat completion via OpenRouter."""
        model = self._get_model(model)
        lc_messages = self._convert_messages(messages)

        try:
            from langchain_openai import ChatOpenAI

            client = ChatOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                streaming=True,
                default_headers={
                    "HTTP-Referer": "https://github.com/aphelion-magnetar",
                    "X-Title": "Aphelion Magnetar",
                },
            )

            input_tokens = 0
            output_tokens = 0

            async for chunk in client.astream(lc_messages, **kwargs):
                content = ""
                if hasattr(chunk, "content") and chunk.content:
                    content = str(chunk.content)

                if hasattr(chunk, "response_metadata"):
                    usage = chunk.response_metadata.get("token_usage", {})
                    if usage:
                        input_tokens = usage.get("prompt_tokens", 0)
                        output_tokens = usage.get("completion_tokens", 0)

                yield StreamChunk(content=content)

            yield StreamChunk(
                content="",
                is_final=True,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )

        except Exception as e:
            logger.exception("OpenRouter stream error", error=str(e), model=model)
            msg = f"OpenRouter stream failed: {e}"
            raise ProviderError(
                msg,
                provider_name=self.name,
            ) from e

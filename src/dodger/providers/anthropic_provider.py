"""Anthropic provider implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

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


# Known Anthropic models
ANTHROPIC_MODELS: dict[str, ModelInfo] = {
    "claude-3-5-sonnet-20241022": ModelInfo(
        id="claude-3-5-sonnet-20241022",
        name="Claude 3.5 Sonnet",
        context_window=200000,
        supports_vision=True,
        supports_function_calling=True,
        input_cost_per_1m=3.00,
        output_cost_per_1m=15.00,
    ),
    "claude-3-5-haiku-20241022": ModelInfo(
        id="claude-3-5-haiku-20241022",
        name="Claude 3.5 Haiku",
        context_window=200000,
        supports_vision=True,
        supports_function_calling=True,
        input_cost_per_1m=0.80,
        output_cost_per_1m=4.00,
    ),
    "claude-3-opus-20240229": ModelInfo(
        id="claude-3-opus-20240229",
        name="Claude 3 Opus",
        context_window=200000,
        supports_vision=True,
        supports_function_calling=True,
        input_cost_per_1m=15.00,
        output_cost_per_1m=75.00,
    ),
    "claude-3-sonnet-20240229": ModelInfo(
        id="claude-3-sonnet-20240229",
        name="Claude 3 Sonnet",
        context_window=200000,
        supports_vision=True,
        supports_function_calling=True,
        input_cost_per_1m=3.00,
        output_cost_per_1m=15.00,
    ),
    "claude-3-haiku-20240307": ModelInfo(
        id="claude-3-haiku-20240307",
        name="Claude 3 Haiku",
        context_window=200000,
        supports_vision=True,
        supports_function_calling=True,
        input_cost_per_1m=0.25,
        output_cost_per_1m=1.25,
    ),
}


class AnthropicProvider(BaseProvider):
    """Anthropic Claude API provider.

    Supports Claude 3.5, Claude 3 Opus, Sonnet, and Haiku models.
    """

    name = "anthropic"
    display_name = "Anthropic"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        default_model: str | None = "claude-3-5-sonnet-20241022",
        **kwargs: Any,
    ) -> None:
        """Initialize the Anthropic provider."""
        super().__init__(
            api_key=api_key,
            base_url=base_url,  # Anthropic SDK handles base URL
            default_model=default_model,
            **kwargs,
        )

    @property
    def capabilities(self) -> ProviderCapabilities:
        """Get Anthropic capabilities."""
        return ProviderCapabilities(
            streaming=True,
            vision=True,
            function_calling=True,
            json_mode=True,
            system_messages=True,
            max_context_window=200000,
        )

    async def validate_connection(self) -> bool:
        """Validate the Anthropic connection."""
        if not self.api_key:
            msg = "Anthropic API key is required"
            raise InvalidAPIKeyError(
                msg,
                provider_name=self.name,
            )

        try:
            from langchain_anthropic import ChatAnthropic
            from langchain_core.messages import HumanMessage

            client = ChatAnthropic(
                api_key=self.api_key,
                model="claude-3-haiku-20240307",  # Cheapest for validation
                max_tokens=5,
            )
            await client.ainvoke([HumanMessage(content="Hi")])
            return True

        except Exception as e:
            error_msg = str(e).lower()
            if "invalid api key" in error_msg or "authentication" in error_msg:
                msg = "Invalid Anthropic API key"
                raise InvalidAPIKeyError(
                    msg,
                    provider_name=self.name,
                ) from e
            msg = f"Failed to connect to Anthropic: {e}"
            raise ConnectionError(
                msg,
                provider_name=self.name,
            ) from e

    async def get_available_models(self) -> list[ModelInfo]:
        """Get available Anthropic models."""
        return list(ANTHROPIC_MODELS.values())

    async def chat(
        self,
        messages: list[ChatMessage],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """Send a chat completion request to Anthropic."""
        model = self._get_model(model)
        lc_messages = self._convert_messages(messages)

        try:
            from langchain_anthropic import ChatAnthropic

            client = ChatAnthropic(
                api_key=self.api_key,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens or 4096,
            )

            response = await client.ainvoke(lc_messages, **kwargs)

            # Extract token usage
            input_tokens = 0
            output_tokens = 0
            if hasattr(response, "response_metadata"):
                usage = response.response_metadata.get("usage", {})
                input_tokens = usage.get("input_tokens", 0)
                output_tokens = usage.get("output_tokens", 0)

            return ChatResponse(
                content=str(response.content),
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                finish_reason=response.response_metadata.get("stop_reason"),
            )

        except Exception as e:
            logger.exception("Anthropic chat error", error=str(e), model=model)
            msg = f"Anthropic chat failed: {e}"
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
        """Stream a chat completion from Anthropic."""
        model = self._get_model(model)
        lc_messages = self._convert_messages(messages)

        try:
            from langchain_anthropic import ChatAnthropic

            client = ChatAnthropic(
                api_key=self.api_key,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens or 4096,
                streaming=True,
            )

            input_tokens = 0
            output_tokens = 0

            async for chunk in client.astream(lc_messages, **kwargs):
                content = ""
                if hasattr(chunk, "content") and chunk.content:
                    content = str(chunk.content)

                # Try to get usage from final chunk
                if hasattr(chunk, "response_metadata"):
                    usage = chunk.response_metadata.get("usage", {})
                    if usage:
                        input_tokens = usage.get("input_tokens", 0)
                        output_tokens = usage.get("output_tokens", 0)

                yield StreamChunk(content=content)

            yield StreamChunk(
                content="",
                is_final=True,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )

        except Exception as e:
            logger.exception("Anthropic stream error", error=str(e), model=model)
            msg = f"Anthropic stream failed: {e}"
            raise ProviderError(
                msg,
                provider_name=self.name,
            ) from e

"""Google Gemini provider implementation."""

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


# Known Google Gemini models
GOOGLE_MODELS: dict[str, ModelInfo] = {
    "gemini-2.0-flash-exp": ModelInfo(
        id="gemini-2.0-flash-exp",
        name="Gemini 2.0 Flash (Experimental)",
        context_window=1000000,
        supports_vision=True,
        supports_function_calling=True,
        input_cost_per_1m=0.0,  # Free tier
        output_cost_per_1m=0.0,
    ),
    "gemini-1.5-pro": ModelInfo(
        id="gemini-1.5-pro",
        name="Gemini 1.5 Pro",
        context_window=2000000,
        supports_vision=True,
        supports_function_calling=True,
        input_cost_per_1m=1.25,
        output_cost_per_1m=5.00,
    ),
    "gemini-1.5-flash": ModelInfo(
        id="gemini-1.5-flash",
        name="Gemini 1.5 Flash",
        context_window=1000000,
        supports_vision=True,
        supports_function_calling=True,
        input_cost_per_1m=0.075,
        output_cost_per_1m=0.30,
    ),
    "gemini-1.5-flash-8b": ModelInfo(
        id="gemini-1.5-flash-8b",
        name="Gemini 1.5 Flash 8B",
        context_window=1000000,
        supports_vision=True,
        supports_function_calling=True,
        input_cost_per_1m=0.0375,
        output_cost_per_1m=0.15,
    ),
}


class GoogleProvider(BaseProvider):
    """Google Gemini API provider.

    Supports Gemini 2.0, 1.5 Pro, and Flash models.
    """

    name = "google"
    display_name = "Google AI"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        default_model: str | None = "gemini-2.0-flash-exp",
        **kwargs: Any,
    ) -> None:
        """Initialize the Google provider."""
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            default_model=default_model,
            **kwargs,
        )

    @property
    def capabilities(self) -> ProviderCapabilities:
        """Get Google capabilities."""
        return ProviderCapabilities(
            streaming=True,
            vision=True,
            function_calling=True,
            json_mode=True,
            system_messages=True,
            max_context_window=2000000,  # Gemini 1.5 Pro has 2M context
        )

    async def validate_connection(self) -> bool:
        """Validate the Google connection."""
        if not self.api_key:
            msg = "Google API key is required"
            raise InvalidAPIKeyError(
                msg,
                provider_name=self.name,
            )

        try:
            from langchain_core.messages import HumanMessage
            from langchain_google_genai import ChatGoogleGenerativeAI

            client = ChatGoogleGenerativeAI(
                google_api_key=self.api_key,
                model="gemini-1.5-flash",  # Use cheaper model for validation
                max_output_tokens=5,
            )
            await client.ainvoke([HumanMessage(content="Hi")])
            return True

        except Exception as e:
            error_msg = str(e).lower()
            if "invalid api key" in error_msg or "api_key" in error_msg:
                msg = "Invalid Google API key"
                raise InvalidAPIKeyError(
                    msg,
                    provider_name=self.name,
                ) from e
            msg = f"Failed to connect to Google AI: {e}"
            raise ConnectionError(
                msg,
                provider_name=self.name,
            ) from e

    async def get_available_models(self) -> list[ModelInfo]:
        """Get available Google models."""
        return list(GOOGLE_MODELS.values())

    async def chat(
        self,
        messages: list[ChatMessage],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """Send a chat completion request to Google."""
        model = self._get_model(model)
        lc_messages = self._convert_messages(messages)

        try:
            from langchain_google_genai import ChatGoogleGenerativeAI

            client = ChatGoogleGenerativeAI(
                google_api_key=self.api_key,
                model=model,
                temperature=temperature,
                max_output_tokens=max_tokens,
            )

            response = await client.ainvoke(lc_messages, **kwargs)

            # Extract token usage
            input_tokens = 0
            output_tokens = 0
            if hasattr(response, "response_metadata"):
                usage = response.response_metadata.get("usage_metadata", {})
                input_tokens = usage.get("prompt_token_count", 0)
                output_tokens = usage.get("candidates_token_count", 0)

            return ChatResponse(
                content=str(response.content),
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                finish_reason=response.response_metadata.get("finish_reason"),
            )

        except Exception as e:
            logger.exception("Google chat error", error=str(e), model=model)
            msg = f"Google AI chat failed: {e}"
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
        """Stream a chat completion from Google."""
        model = self._get_model(model)
        lc_messages = self._convert_messages(messages)

        try:
            from langchain_google_genai import ChatGoogleGenerativeAI

            client = ChatGoogleGenerativeAI(
                google_api_key=self.api_key,
                model=model,
                temperature=temperature,
                max_output_tokens=max_tokens,
                streaming=True,
            )

            input_tokens = 0
            output_tokens = 0

            async for chunk in client.astream(lc_messages, **kwargs):
                content = ""
                if hasattr(chunk, "content") and chunk.content:
                    content = str(chunk.content)

                # Try to get usage from chunks
                if hasattr(chunk, "response_metadata"):
                    usage = chunk.response_metadata.get("usage_metadata", {})
                    if usage:
                        input_tokens = usage.get("prompt_token_count", 0)
                        output_tokens = usage.get("candidates_token_count", 0)

                yield StreamChunk(content=content)

            yield StreamChunk(
                content="",
                is_final=True,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )

        except Exception as e:
            logger.exception("Google stream error", error=str(e), model=model)
            msg = f"Google AI stream failed: {e}"
            raise ProviderError(
                msg,
                provider_name=self.name,
            ) from e

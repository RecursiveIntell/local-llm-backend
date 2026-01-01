"""LMStudio provider implementation for local LLM inference.

LMStudio provides an OpenAI-compatible API for local models.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import httpx
import structlog

from dodger.core.exceptions import ConnectionError, ProviderError
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


class LMStudioProvider(BaseProvider):
    """LMStudio provider for local LLM inference.

    LMStudio provides an OpenAI-compatible API, making it easy
    to use local models with the same interface as cloud APIs.
    """

    name = "lmstudio"
    display_name = "LM Studio (Local)"

    def __init__(
        self,
        api_key: str | None = None,  # Not used
        base_url: str | None = None,
        default_model: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the LMStudio provider."""
        super().__init__(
            api_key=api_key or "lm-studio",  # Placeholder, not validated
            base_url=base_url or "http://localhost:1234/v1",
            default_model=default_model,
            **kwargs,
        )

    @property
    def capabilities(self) -> ProviderCapabilities:
        """Get LMStudio capabilities."""
        return ProviderCapabilities(
            streaming=True,
            vision=False,  # Most local models don't support vision
            function_calling=False,
            json_mode=True,
            system_messages=True,
            max_context_window=32000,  # Varies by model
        )

    async def validate_connection(self) -> bool:
        """Validate that LMStudio is running."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/models")
                if response.status_code == 200:
                    logger.info("LMStudio connection validated", base_url=self.base_url)
                    return True
                else:
                    logger.warning(
                        "LMStudio returned non-200",
                        base_url=self.base_url,
                        status=response.status_code,
                    )
                    return False

        except httpx.TimeoutException:
            logger.warning("LMStudio connection timeout", base_url=self.base_url)
            return False
        except httpx.ConnectError as e:
            logger.warning("LMStudio not reachable", base_url=self.base_url, error=str(e))
            return False
        except Exception as e:
            logger.warning("LMStudio validation error", base_url=self.base_url, error=str(e))
            return False

    async def get_available_models(self) -> list[ModelInfo]:
        """Get loaded models from LMStudio."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{self.base_url}/models")
                response.raise_for_status()
                data = response.json()

            models: list[ModelInfo] = []
            for model_data in data.get("data", []):
                model_id = model_data.get("id", "")

                models.append(
                    ModelInfo(
                        id=model_id,
                        name=model_id,
                        context_window=model_data.get("context_length", 4096),
                        supports_vision=False,
                        supports_function_calling=False,
                        input_cost_per_1m=0.0,  # Free (local)
                        output_cost_per_1m=0.0,
                    )
                )

            return models

        except Exception as e:
            logger.warning("Failed to fetch LMStudio models", error=str(e))
            return []

    async def chat(
        self,
        messages: list[ChatMessage],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """Send a chat completion request to LMStudio."""
        model = self._get_model(model)
        lc_messages = self._convert_messages(messages)

        try:
            from langchain_openai import ChatOpenAI

            # LMStudio uses OpenAI-compatible API
            client = ChatOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
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
            logger.exception("LMStudio chat error", error=str(e), model=model)
            msg = f"LMStudio chat failed: {e}"
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
        """Stream a chat completion from LMStudio."""
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
            logger.exception("LMStudio stream error", error=str(e), model=model)
            msg = f"LMStudio stream failed: {e}"
            raise ProviderError(
                msg,
                provider_name=self.name,
            ) from e

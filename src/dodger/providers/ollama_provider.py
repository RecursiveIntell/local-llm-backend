"""Ollama provider implementation for local LLM inference."""

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


class OllamaProvider(BaseProvider):
    """Ollama provider for local LLM inference.

    Ollama runs on the local machine and provides access to
    open-source models like Llama, Mistral, CodeLlama, etc.
    """

    name = "ollama"
    display_name = "Ollama (Local)"

    def __init__(
        self,
        api_key: str | None = None,  # Not used, but kept for interface consistency
        base_url: str | None = None,
        default_model: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the Ollama provider."""
        super().__init__(
            api_key=api_key,
            base_url=base_url or "http://localhost:11434",
            default_model=default_model,
            **kwargs,
        )

    @property
    def capabilities(self) -> ProviderCapabilities:
        """Get Ollama capabilities."""
        return ProviderCapabilities(
            streaming=True,
            vision=True,  # Some models support vision (llava, etc.)
            function_calling=False,  # Limited support
            json_mode=True,
            system_messages=True,
            max_context_window=128000,  # Varies by model
        )

    async def validate_connection(self) -> bool:
        """Validate that Ollama is running."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                if response.status_code == 200:
                    logger.info("Ollama connection validated", base_url=self.base_url)
                    return True
                else:
                    logger.warning("Ollama returned non-200", status=response.status_code)
                    return False

        except httpx.TimeoutException:
            logger.warning("Ollama connection timeout", base_url=self.base_url)
            return False
        except httpx.ConnectError as e:
            logger.warning("Ollama not reachable", base_url=self.base_url, error=str(e))
            return False
        except Exception as e:
            logger.warning("Ollama validation error", base_url=self.base_url, error=str(e))
            return False

    async def get_available_models(self) -> list[ModelInfo]:
        """Get installed Ollama models."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                response.raise_for_status()
                data = response.json()

            models: list[ModelInfo] = []
            for model_data in data.get("models", []):
                name = model_data.get("name", "")
                details = model_data.get("details", {})

                # Determine capabilities from model name
                supports_vision = any(
                    v in name.lower() for v in ["llava", "bakllava", "vision"]
                )

                models.append(
                    ModelInfo(
                        id=name,
                        name=name,
                        context_window=details.get("context_length", 4096),
                        supports_vision=supports_vision,
                        supports_function_calling=False,
                        input_cost_per_1m=0.0,  # Free (local)
                        output_cost_per_1m=0.0,
                    )
                )

            return models

        except Exception as e:
            logger.warning("Failed to fetch Ollama models", error=str(e))
            return []

    async def pull_model(self, model_name: str) -> AsyncGenerator[str, None]:
        """Pull (download) a model from Ollama registry.

        Args:
            model_name: Name of the model to pull.

        Yields:
            Status updates during download.
        """
        try:
            async with httpx.AsyncClient(timeout=None) as client:  # No timeout for pulls
                async with client.stream(
                    "POST",
                    f"{self.base_url}/api/pull",
                    json={"name": model_name, "stream": True},
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if line:
                            import json

                            data = json.loads(line)
                            status = data.get("status", "")
                            if status:
                                yield status

        except Exception as e:
            msg = f"Failed to pull model {model_name}: {e}"
            raise ProviderError(
                msg,
                provider_name=self.name,
            ) from e

    async def chat(
        self,
        messages: list[ChatMessage],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """Send a chat completion request to Ollama."""
        model = self._get_model(model)

        try:
            from langchain_community.chat_models import ChatOllama

            client = ChatOllama(
                base_url=self.base_url,
                model=model,
                temperature=temperature,
                num_predict=max_tokens,
            )

            lc_messages = self._convert_messages(messages)
            response = await client.ainvoke(lc_messages, **kwargs)

            # Ollama doesn't always provide token counts
            return ChatResponse(
                content=str(response.content),
                model=model,
                input_tokens=0,
                output_tokens=0,
            )

        except Exception as e:
            logger.exception("Ollama chat error", error=str(e), model=model)
            msg = f"Ollama chat failed: {e}"
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
        """Stream a chat completion from Ollama."""
        model = self._get_model(model)

        try:
            from langchain_community.chat_models import ChatOllama

            client = ChatOllama(
                base_url=self.base_url,
                model=model,
                temperature=temperature,
                num_predict=max_tokens,
                streaming=True,
            )

            lc_messages = self._convert_messages(messages)

            async for chunk in client.astream(lc_messages, **kwargs):
                content = ""
                if hasattr(chunk, "content") and chunk.content:
                    content = str(chunk.content)
                yield StreamChunk(content=content)

            yield StreamChunk(content="", is_final=True)

        except Exception as e:
            logger.exception("Ollama stream error", error=str(e), model=model)
            msg = f"Ollama stream failed: {e}"
            raise ProviderError(
                msg,
                provider_name=self.name,
            ) from e

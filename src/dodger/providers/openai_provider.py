"""OpenAI provider implementation."""

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


# Known OpenAI models with their capabilities
OPENAI_MODELS: dict[str, ModelInfo] = {
    "gpt-4o": ModelInfo(
        id="gpt-4o",
        name="GPT-4o",
        context_window=128000,
        supports_vision=True,
        supports_function_calling=True,
        input_cost_per_1m=2.50,
        output_cost_per_1m=10.00,
    ),
    "gpt-4o-mini": ModelInfo(
        id="gpt-4o-mini",
        name="GPT-4o Mini",
        context_window=128000,
        supports_vision=True,
        supports_function_calling=True,
        input_cost_per_1m=0.15,
        output_cost_per_1m=0.60,
    ),
    "gpt-4-turbo": ModelInfo(
        id="gpt-4-turbo",
        name="GPT-4 Turbo",
        context_window=128000,
        supports_vision=True,
        supports_function_calling=True,
        input_cost_per_1m=10.00,
        output_cost_per_1m=30.00,
    ),
    "gpt-3.5-turbo": ModelInfo(
        id="gpt-3.5-turbo",
        name="GPT-3.5 Turbo",
        context_window=16385,
        supports_vision=False,
        supports_function_calling=True,
        input_cost_per_1m=0.50,
        output_cost_per_1m=1.50,
    ),
    "o1": ModelInfo(
        id="o1",
        name="o1",
        context_window=200000,
        supports_vision=True,
        supports_function_calling=True,
        input_cost_per_1m=15.00,
        output_cost_per_1m=60.00,
    ),
    "o1-mini": ModelInfo(
        id="o1-mini",
        name="o1 Mini",
        context_window=128000,
        supports_vision=True,
        supports_function_calling=True,
        input_cost_per_1m=3.00,
        output_cost_per_1m=12.00,
    ),
}


class OpenAIProvider(BaseProvider):
    """OpenAI API provider.

    Supports GPT-4o, GPT-4, GPT-3.5, and o1 models.
    """

    name = "openai"
    display_name = "OpenAI"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        default_model: str | None = "gpt-4o",
        **kwargs: Any,
    ) -> None:
        """Initialize the OpenAI provider."""
        super().__init__(
            api_key=api_key,
            base_url=base_url or "https://api.openai.com/v1",
            default_model=default_model,
            **kwargs,
        )
        self._client: Any = None

    def _get_client(self) -> Any:
        """Get or create the LangChain ChatOpenAI client."""
        if self._client is None:
            from langchain_openai import ChatOpenAI

            self._client = ChatOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                model=self.default_model or "gpt-4o",
                streaming=True,
            )
        return self._client

    @property
    def capabilities(self) -> ProviderCapabilities:
        """Get OpenAI capabilities."""
        return ProviderCapabilities(
            streaming=True,
            vision=True,
            function_calling=True,
            json_mode=True,
            system_messages=True,
            max_context_window=128000,
        )

    async def validate_connection(self) -> bool:
        """Validate the OpenAI connection."""
        if not self.api_key:
            msg = "OpenAI API key is required"
            raise InvalidAPIKeyError(
                msg,
                provider_name=self.name,
            )

        try:
            # Try a minimal API call
            client = self._get_client()
            from langchain_core.messages import HumanMessage

            # Use a cheap model for validation
            test_client = client.bind(model="gpt-3.5-turbo", max_tokens=5)
            await test_client.ainvoke([HumanMessage(content="Hi")])
            return True

        except Exception as e:
            error_msg = str(e).lower()
            if "invalid api key" in error_msg or "authentication" in error_msg:
                msg = "Invalid OpenAI API key"
                raise InvalidAPIKeyError(
                    msg,
                    provider_name=self.name,
                ) from e
            msg = f"Failed to connect to OpenAI: {e}"
            raise ConnectionError(
                msg,
                provider_name=self.name,
            ) from e

    async def get_available_models(self) -> list[ModelInfo]:
        """Get available OpenAI models from API."""
        try:
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/models",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    timeout=30.0,
                )
                response.raise_for_status()
                data = response.json()

            models = []
            for model_data in data.get("data", []):
                model_id = model_data.get("id", "")
                # Only include chat models (gpt, o1, chatgpt)
                if any(x in model_id.lower() for x in ["gpt", "o1", "chatgpt"]):
                    # Skip fine-tuned, instruct-only, and deprecated models
                    if any(x in model_id for x in [":ft-", "instruct", "0301", "0314", "0613", "1106", "0125"]):
                        continue
                    # Get known info or create basic info
                    known = OPENAI_MODELS.get(model_id)
                    if known:
                        models.append(known)
                    else:
                        models.append(ModelInfo(
                            id=model_id,
                            name=model_id,
                            context_window=128000,
                            supports_vision="vision" in model_id or "4o" in model_id or model_id.startswith("o1"),
                            supports_function_calling=True,
                        ))

            # Sort by name
            models.sort(key=lambda m: m.name)
            logger.info("Fetched OpenAI models from API", count=len(models))
            return models if models else list(OPENAI_MODELS.values())

        except Exception as e:
            logger.warning("Failed to fetch models from API, using static list", error=str(e))
            return list(OPENAI_MODELS.values())

    async def chat(
        self,
        messages: list[ChatMessage],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """Send a chat completion request to OpenAI."""
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
                streaming=False,
            )

            response = await client.ainvoke(lc_messages, **kwargs)

            # Extract token usage
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
                finish_reason=response.response_metadata.get("finish_reason"),
            )

        except Exception as e:
            logger.exception("OpenAI chat error", error=str(e), model=model)
            msg = f"OpenAI chat failed: {e}"
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
        """Stream a chat completion from OpenAI."""
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

                # Try to get usage from final chunk
                if hasattr(chunk, "response_metadata"):
                    usage = chunk.response_metadata.get("token_usage", {})
                    if usage:
                        input_tokens = usage.get("prompt_tokens", 0)
                        output_tokens = usage.get("completion_tokens", 0)

                yield StreamChunk(content=content)

            # Final chunk with token counts
            yield StreamChunk(
                content="",
                is_final=True,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )

        except Exception as e:
            logger.exception("OpenAI stream error", error=str(e), model=model)
            msg = f"OpenAI stream failed: {e}"
            raise ProviderError(
                msg,
                provider_name=self.name,
            ) from e

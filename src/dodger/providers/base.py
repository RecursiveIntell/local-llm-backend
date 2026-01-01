"""Abstract base class for LLM providers.

This module defines the interface that all LLM providers must implement,
ensuring consistent behavior across different backends.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from langchain_core.messages import BaseMessage


@dataclass
class ProviderCapabilities:
    """Capabilities supported by a provider.

    Attributes:
        streaming: Whether the provider supports streaming responses.
        vision: Whether the provider supports image inputs.
        function_calling: Whether the provider supports function/tool calling.
        json_mode: Whether the provider supports structured JSON output.
        system_messages: Whether the provider supports system messages.
        max_context_window: Maximum context window size in tokens.
    """

    streaming: bool = True
    vision: bool = False
    function_calling: bool = False
    json_mode: bool = False
    system_messages: bool = True
    max_context_window: int = 128000


@dataclass
class ModelInfo:
    """Information about a model.

    Attributes:
        id: Model identifier.
        name: Human-friendly name.
        context_window: Context window size.
        supports_vision: Whether it supports image inputs.
        supports_function_calling: Whether it supports tools.
        input_cost_per_1m: Cost per 1M input tokens.
        output_cost_per_1m: Cost per 1M output tokens.
    """

    id: str
    name: str
    context_window: int = 128000
    supports_vision: bool = False
    supports_function_calling: bool = False
    input_cost_per_1m: float = 0.0
    output_cost_per_1m: float = 0.0


@dataclass
class ChatResponse:
    """Response from a chat completion.

    Attributes:
        content: The generated text content.
        model: The model that generated the response.
        input_tokens: Number of input tokens used.
        output_tokens: Number of output tokens generated.
        finish_reason: Why the generation stopped.
        metadata: Additional provider-specific metadata.
    """

    content: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    finish_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        """Get total token count."""
        return self.input_tokens + self.output_tokens


@dataclass
class StreamChunk:
    """A chunk of a streaming response.

    Attributes:
        content: The text content of this chunk.
        is_final: Whether this is the final chunk.
        input_tokens: Input tokens (only in final chunk).
        output_tokens: Output tokens (only in final chunk).
    """

    content: str
    is_final: bool = False
    input_tokens: int = 0
    output_tokens: int = 0


class ChatMessage(BaseModel):
    """A chat message for provider input.

    This is a simplified message format that providers convert
    to their specific format internally.
    """

    role: str  # user, assistant, system
    content: str
    images: list[str] = []  # Base64 encoded images or URLs


class BaseProvider(ABC):
    """Abstract base class for LLM providers.

    All provider implementations must inherit from this class
    and implement the abstract methods.
    """

    # Provider identification
    name: str = "base"
    display_name: str = "Base Provider"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        default_model: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the provider.

        Args:
            api_key: API key for authentication.
            base_url: Base URL for the API.
            default_model: Default model to use.
            **kwargs: Additional provider-specific options.
        """
        self.api_key = api_key
        self.base_url = base_url
        self.default_model = default_model
        self._kwargs = kwargs

    @property
    @abstractmethod
    def capabilities(self) -> ProviderCapabilities:
        """Get the capabilities of this provider.

        Returns:
            The provider's capabilities.
        """
        ...

    @abstractmethod
    async def validate_connection(self) -> bool:
        """Validate that the provider connection works.

        Returns:
            True if the connection is valid.

        Raises:
            ConnectionError: If connection fails.
            InvalidAPIKeyError: If the API key is invalid.
        """
        ...

    @abstractmethod
    async def get_available_models(self) -> list[ModelInfo]:
        """Get available models from this provider.

        Returns:
            List of available models.
        """
        ...

    @abstractmethod
    async def chat(
        self,
        messages: list[ChatMessage],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """Send a chat completion request.

        Args:
            messages: List of messages in the conversation.
            model: Model to use (defaults to default_model).
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            **kwargs: Additional provider-specific options.

        Returns:
            The chat response.

        Raises:
            ProviderError: If the request fails.
        """
        ...

    @abstractmethod
    async def stream_chat(
        self,
        messages: list[ChatMessage],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[StreamChunk, None]:
        """Stream a chat completion response.

        Args:
            messages: List of messages in the conversation.
            model: Model to use (defaults to default_model).
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            **kwargs: Additional provider-specific options.

        Yields:
            Chunks of the response as they're generated.

        Raises:
            ProviderError: If the request fails.
        """
        ...

    def _get_model(self, model: str | None) -> str:
        """Get the model to use, with fallback to default.

        Args:
            model: The requested model, or None.

        Returns:
            The model to use.

        Raises:
            ValueError: If no model is specified and no default is set.
        """
        model = model or self.default_model
        if not model:
            msg = f"No model specified and no default model set for {self.name}"
            raise ValueError(msg)
        return model

    def _convert_messages(self, messages: list[ChatMessage]) -> list[BaseMessage]:
        """Convert ChatMessage to LangChain messages.

        Args:
            messages: List of ChatMessage objects.

        Returns:
            List of LangChain message objects.
        """
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

        result: list[BaseMessage] = []
        for msg in messages:
            if msg.role == "user":
                if msg.images:
                    # Vision message with images
                    content: list[dict[str, Any]] = [{"type": "text", "text": msg.content}]
                    for image in msg.images:
                        if image.startswith("data:"):
                            # Base64 encoded
                            content.append({
                                "type": "image_url",
                                "image_url": {"url": image},
                            })
                        else:
                            # URL
                            content.append({
                                "type": "image_url",
                                "image_url": {"url": image},
                            })
                    result.append(HumanMessage(content=content))
                else:
                    result.append(HumanMessage(content=msg.content))
            elif msg.role == "assistant":
                result.append(AIMessage(content=msg.content))
            elif msg.role == "system":
                result.append(SystemMessage(content=msg.content))
        return result

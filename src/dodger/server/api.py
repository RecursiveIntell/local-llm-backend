"""FastAPI server for remote LLM access.

This server enables mobile devices to connect to your local LLMs
or use your configured cloud APIs remotely.

Features:
- REST endpoints for chat
- WebSocket for streaming
- Token-based authentication
- CORS for mobile apps
"""

from __future__ import annotations

import secrets
from datetime import UTC, datetime

import structlog
import uvicorn
from fastapi import Depends, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from dodger.config import ProviderType, config_manager
from dodger.core.conversation import conversation_manager
from dodger.providers.base import ChatMessage
from dodger.providers.manager import provider_manager

logger = structlog.get_logger(__name__)

# FastAPI app
app = FastAPI(
    title="Aphelion Magnetar API",
    description="Remote access to your LLM chat application",
    version="0.1.0",
)

# Security
security = HTTPBearer()


def get_api_token() -> str | None:
    """Get the configured API token."""
    settings = config_manager.settings
    if settings.server.api_token:
        return settings.server.api_token.get_secret_value()
    return None


async def verify_token(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> str:
    """Verify the API token."""
    expected_token = get_api_token()

    if not expected_token:
        raise HTTPException(
            status_code=503,
            detail="Server not configured. Please set an API token in settings.",
        )

    if credentials.credentials != expected_token:
        raise HTTPException(
            status_code=401,
            detail="Invalid API token",
        )

    return credentials.credentials


# CORS configuration
def configure_cors() -> None:
    """Configure CORS for the app."""
    settings = config_manager.settings
    origins = settings.server.cors_origins

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


configure_cors()


# Request/Response models
class ChatRequest(BaseModel):
    """Chat completion request."""

    messages: list[dict] = Field(..., description="List of messages")
    model: str | None = Field(None, description="Model to use")
    temperature: float = Field(0.7, ge=0, le=2)
    max_tokens: int | None = Field(None, ge=1)
    stream: bool = Field(False, description="Whether to stream the response")


class ChatResponse(BaseModel):
    """Chat completion response."""

    content: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    created_at: str


class ConversationSummary(BaseModel):
    """Summary of a conversation."""

    id: str
    title: str
    preview: str
    updated_at: str
    message_count: int


class ModelInfo(BaseModel):
    """Information about a model."""

    id: str
    name: str
    provider: str
    context_window: int = 0
    supports_vision: bool = False


class ServerStatus(BaseModel):
    """Server status information."""

    status: str
    version: str
    active_provider: str
    active_model: str | None
    uptime_seconds: float


class SearchRequest(BaseModel):
    """Web search request."""

    query: str = Field(..., description="Search query")
    max_results: int = Field(5, ge=1, le=10, description="Maximum results to return")


class SearchResultItem(BaseModel):
    """A single search result."""

    title: str
    url: str
    snippet: str


class SearchResponse(BaseModel):
    """Web search response."""

    query: str
    results: list[SearchResultItem]
    error: str | None = None


class ProviderInfo(BaseModel):
    """Provider configuration info."""

    name: str
    display_name: str
    enabled: bool
    has_api_key: bool
    base_url: str | None
    default_model: str | None
    available_models: list[str]
    requires_api_key: bool
    description: str


class ProviderUpdateRequest(BaseModel):
    """Request to update a provider."""

    enabled: bool | None = None
    api_key: str | None = None
    base_url: str | None = None
    default_model: str | None = None


class ActiveConfigResponse(BaseModel):
    """Active provider and model configuration."""

    provider: str
    model: str | None


class ActiveConfigRequest(BaseModel):
    """Request to set active provider/model."""

    provider: str
    model: str | None = None


class ProviderTestResult(BaseModel):
    """Result of provider connection test."""

    success: bool
    message: str


class ProviderModelsResponse(BaseModel):
    """Response with provider models."""

    provider: str
    models: list[ModelInfo]


# Track server start time
_server_start_time = datetime.now(UTC)


# Routes
@app.get("/")
async def root() -> dict:
    """Root endpoint."""
    return {"message": "Aphelion Magnetar API", "version": "0.1.0"}


@app.get("/status", response_model=ServerStatus)
async def get_status() -> ServerStatus:
    """Get server status."""
    settings = config_manager.settings
    uptime = (datetime.now(UTC) - _server_start_time).total_seconds()

    return ServerStatus(
        status="running",
        version="0.1.0",
        active_provider=settings.active_provider.value,
        active_model=settings.active_model,
        uptime_seconds=uptime,
    )


@app.get("/models", response_model=list[ModelInfo])
async def get_models(token: str = Depends(verify_token)) -> list[ModelInfo]:
    """Get available models."""
    all_models = await provider_manager.get_all_available_models()

    result: list[ModelInfo] = []
    for provider_type, models in all_models.items():
        for model in models:
            result.append(
                ModelInfo(
                    id=model.id,
                    name=model.name,
                    provider=provider_type.value,
                    context_window=model.context_window,
                    supports_vision=model.supports_vision,
                )
            )

    return result


@app.get("/conversations", response_model=list[ConversationSummary])
async def get_conversations(
    limit: int = 50,
    token: str = Depends(verify_token),
) -> list[ConversationSummary]:
    """Get recent conversations."""
    conversations = await conversation_manager.get_recent_conversations(limit)

    return [
        ConversationSummary(
            id=conv["id"],
            title=conv["title"],
            preview=conv.get("preview", ""),
            updated_at=conv["updated_at"],
            message_count=conv.get("message_count", 0),
        )
        for conv in conversations
    ]


@app.post("/search", response_model=SearchResponse)
async def search_web(
    request: SearchRequest,
    token: str = Depends(verify_token),
) -> SearchResponse:
    """Perform a web search using DuckDuckGo."""
    try:
        from dodger.core.web_search import web_search

        response = await web_search.search_async(request.query, request.max_results)

        return SearchResponse(
            query=response.query,
            results=[
                SearchResultItem(
                    title=r.title,
                    url=r.url,
                    snippet=r.snippet,
                )
                for r in response.results
            ],
            error=response.error,
        )
    except Exception as e:
        logger.exception("Web search failed", query=request.query)
        return SearchResponse(
            query=request.query,
            results=[],
            error=str(e),
        )

@app.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    token: str = Depends(verify_token),
) -> ChatResponse:
    """Send a chat message and get a response."""
    # Convert messages
    messages = [
        ChatMessage(
            role=msg.get("role", "user"),
            content=msg.get("content", ""),
            images=msg.get("images", []),
        )
        for msg in request.messages
    ]

    # Get response
    response = await provider_manager.chat(
        messages=messages,
        model=request.model,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
    )

    return ChatResponse(
        content=response.content,
        model=response.model,
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
        created_at=datetime.now(UTC).isoformat(),
    )


@app.websocket("/chat/stream")
async def chat_stream(websocket: WebSocket) -> None:
    """WebSocket endpoint for streaming chat."""
    await websocket.accept()

    # Verify token from query params or first message
    token = websocket.query_params.get("token")
    expected_token = get_api_token()

    if not expected_token:
        await websocket.close(code=1008, reason="Server not configured")
        return

    if token != expected_token:
        await websocket.close(code=1008, reason="Invalid token")
        return

    try:
        while True:
            # Receive message
            data = await websocket.receive_json()

            messages = [
                ChatMessage(
                    role=msg.get("role", "user"),
                    content=msg.get("content", ""),
                    images=msg.get("images", []),
                )
                for msg in data.get("messages", [])
            ]

            model = data.get("model")
            temperature = data.get("temperature", 0.7)
            max_tokens = data.get("max_tokens")

            # Stream response
            async for chunk in provider_manager.stream_chat(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            ):
                await websocket.send_json({
                    "type": "chunk",
                    "content": chunk.content,
                    "is_final": chunk.is_final,
                    "input_tokens": chunk.input_tokens,
                    "output_tokens": chunk.output_tokens,
                })

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.exception("WebSocket error", error=str(e))
        await websocket.close(code=1011, reason=str(e))


@app.post("/token/generate")
async def generate_token(token: str = Depends(verify_token)) -> dict:
    """Generate a new API token (requires existing valid token)."""
    new_token = secrets.token_urlsafe(32)

    # Update config with new token
    from pydantic import SecretStr

    settings = config_manager.settings
    settings.server.api_token = SecretStr(new_token)
    config_manager.save()

    return {"token": new_token, "note": "Save this token securely. It won't be shown again."}


# Provider info lookup
PROVIDER_INFO = {
    ProviderType.OPENAI: {
        "display_name": "OpenAI",
        "description": "Access GPT-4o, GPT-4, and GPT-3.5 models. Requires an API key from platform.openai.com.",
        "requires_api_key": True,
    },
    ProviderType.ANTHROPIC: {
        "display_name": "Anthropic",
        "description": "Access Claude 3.5 Sonnet, Opus, and Haiku models. Get your API key from console.anthropic.com.",
        "requires_api_key": True,
    },
    ProviderType.GOOGLE: {
        "display_name": "Google AI",
        "description": "Access Gemini 2.0, 1.5 Pro, and Flash models. Get your API key from aistudio.google.com.",
        "requires_api_key": True,
    },
    ProviderType.OPENROUTER: {
        "display_name": "OpenRouter",
        "description": "Unified access to many models (OpenAI, Anthropic, Google, Meta, Mistral) through a single API.",
        "requires_api_key": True,
    },
    ProviderType.OLLAMA: {
        "display_name": "Ollama",
        "description": "Connect to Ollama server (local or remote). Enter IP/hostname in Base URL for remote access.",
        "requires_api_key": False,
    },
    ProviderType.LMSTUDIO: {
        "display_name": "LM Studio",
        "description": "Connect to LM Studio's OpenAI-compatible API. For remote access, enter the server IP/hostname.",
        "requires_api_key": False,
    },
}


@app.get("/providers", response_model=list[ProviderInfo])
async def get_providers(token: str = Depends(verify_token)) -> list[ProviderInfo]:
    """Get all provider configurations."""
    settings = config_manager.settings
    result = []

    for provider_type in ProviderType:
        config = settings.providers[provider_type]
        info = PROVIDER_INFO.get(provider_type, {})

        result.append(
            ProviderInfo(
                name=provider_type.value,
                display_name=info.get("display_name", provider_type.value.title()),
                enabled=config.enabled,
                has_api_key=config.api_key is not None,
                base_url=config.base_url,
                default_model=config.default_model,
                available_models=config.available_models,
                requires_api_key=info.get("requires_api_key", True),
                description=info.get("description", ""),
            )
        )

    return result


@app.get("/providers/{provider_name}", response_model=ProviderInfo)
async def get_provider(
    provider_name: str, token: str = Depends(verify_token)
) -> ProviderInfo:
    """Get a specific provider configuration."""
    try:
        provider_type = ProviderType(provider_name.lower())
    except ValueError:
        raise HTTPException(status_code=404, detail=f"Provider '{provider_name}' not found")

    settings = config_manager.settings
    config = settings.providers[provider_type]
    info = PROVIDER_INFO.get(provider_type, {})

    return ProviderInfo(
        name=provider_type.value,
        display_name=info.get("display_name", provider_type.value.title()),
        enabled=config.enabled,
        has_api_key=config.api_key is not None,
        base_url=config.base_url,
        default_model=config.default_model,
        available_models=config.available_models,
        requires_api_key=info.get("requires_api_key", True),
        description=info.get("description", ""),
    )


@app.put("/providers/{provider_name}", response_model=ProviderInfo)
async def update_provider(
    provider_name: str,
    request: ProviderUpdateRequest,
    token: str = Depends(verify_token),
) -> ProviderInfo:
    """Update a provider's configuration."""
    try:
        provider_type = ProviderType(provider_name.lower())
    except ValueError:
        raise HTTPException(status_code=404, detail=f"Provider '{provider_name}' not found")

    # Update provider config
    config_manager.update_provider(
        provider_type,
        enabled=request.enabled,
        api_key=request.api_key,
        base_url=request.base_url,
        default_model=request.default_model,
    )

    # Return updated config
    return await get_provider(provider_name, token)


@app.get("/providers/{provider_name}/models", response_model=ProviderModelsResponse)
async def get_provider_models(
    provider_name: str, token: str = Depends(verify_token)
) -> ProviderModelsResponse:
    """Get models for a specific provider (refreshes from API)."""
    try:
        provider_type = ProviderType(provider_name.lower())
    except ValueError:
        raise HTTPException(status_code=404, detail=f"Provider '{provider_name}' not found")

    try:
        models = await provider_manager.refresh_models(provider_type)
        return ProviderModelsResponse(
            provider=provider_type.value,
            models=[
                ModelInfo(
                    id=m.id,
                    name=m.name,
                    provider=provider_type.value,
                    context_window=m.context_window,
                    supports_vision=m.supports_vision,
                )
                for m in models
            ],
        )
    except Exception as e:
        logger.error("Failed to get provider models", provider=provider_name, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get models: {e}")


@app.post("/providers/{provider_name}/test", response_model=ProviderTestResult)
async def test_provider(
    provider_name: str, token: str = Depends(verify_token)
) -> ProviderTestResult:
    """Test connection to a provider."""
    try:
        provider_type = ProviderType(provider_name.lower())
    except ValueError:
        raise HTTPException(status_code=404, detail=f"Provider '{provider_name}' not found")

    try:
        # Invalidate cache and test
        provider_manager.invalidate_provider(provider_type)
        success = await provider_manager.validate_provider(provider_type)

        if success:
            return ProviderTestResult(success=True, message="Connection successful!")
        else:
            return ProviderTestResult(success=False, message="Connection failed")
    except Exception as e:
        return ProviderTestResult(success=False, message=str(e))


@app.post("/providers/{provider_name}/detect", response_model=ProviderTestResult)
async def detect_provider(
    provider_name: str, token: str = Depends(verify_token)
) -> ProviderTestResult:
    """Auto-detect local LLM provider (Ollama/LMStudio)."""
    try:
        provider_type = ProviderType(provider_name.lower())
    except ValueError:
        raise HTTPException(status_code=404, detail=f"Provider '{provider_name}' not found")

    if provider_type not in {ProviderType.OLLAMA, ProviderType.LMSTUDIO}:
        raise HTTPException(
            status_code=400,
            detail="Auto-detect only available for Ollama and LMStudio",
        )

    try:
        results = await config_manager.auto_detect_local_llms()
        if results.get(provider_type):
            return ProviderTestResult(
                success=True,
                message=f"{provider_type.value.title()} detected and configured!",
            )
        else:
            return ProviderTestResult(
                success=False,
                message=f"{provider_type.value.title()} not running or not accessible.",
            )
    except Exception as e:
        return ProviderTestResult(success=False, message=str(e))


@app.get("/config/active", response_model=ActiveConfigResponse)
async def get_active_config(token: str = Depends(verify_token)) -> ActiveConfigResponse:
    """Get the active provider and model."""
    settings = config_manager.settings
    return ActiveConfigResponse(
        provider=settings.active_provider.value,
        model=settings.active_model,
    )


@app.put("/config/active", response_model=ActiveConfigResponse)
async def set_active_config(
    request: ActiveConfigRequest, token: str = Depends(verify_token)
) -> ActiveConfigResponse:
    """Set the active provider and model."""
    try:
        provider_type = ProviderType(request.provider.lower())
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid provider '{request.provider}'")

    settings = config_manager.settings

    # Check provider is enabled
    if not settings.providers[provider_type].enabled:
        raise HTTPException(
            status_code=400,
            detail=f"Provider '{request.provider}' is not enabled",
        )

    settings.active_provider = provider_type
    settings.active_model = request.model
    config_manager.save()

    return ActiveConfigResponse(
        provider=settings.active_provider.value,
        model=settings.active_model,
    )


def run_server(
    host: str = "0.0.0.0",  # noqa: S104
    port: int = 8000,
) -> None:
    """Run the API server.

    Args:
        host: Host to bind to.
        port: Port to bind to.
    """
    # Load config
    config_manager.load()

    settings = config_manager.settings
    host = settings.server.host or host
    port = settings.server.port or port

    logger.info("Starting Aphelion API server", host=host, port=port)

    # Check if token is configured
    if not get_api_token():
        # Generate initial token
        initial_token = secrets.token_urlsafe(32)
        from pydantic import SecretStr

        settings.server.api_token = SecretStr(initial_token)
        settings.server.enabled = True
        config_manager.save()

        print("\n" + "=" * 60)
        print("INITIAL API TOKEN (save this securely):")
        print(f"  {initial_token}")
        print("=" * 60 + "\n")

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
    )


if __name__ == "__main__":
    run_server()

"""LangSmith telemetry and observability integration.

This module provides:
- LangSmith tracing for LLM calls
- Token usage tracking
- Cost estimation
- Local fallback logging when LangSmith is unavailable
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

import structlog

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Generator

    from langchain_core.callbacks import BaseCallbackHandler

logger = structlog.get_logger(__name__)


# Token pricing per 1M tokens (as of Dec 2024)
TOKEN_PRICING: dict[str, dict[str, float]] = {
    # OpenAI
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    # Anthropic
    "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
    "claude-3-opus-20240229": {"input": 15.00, "output": 75.00},
    "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
    # Google
    "gemini-2.0-flash-exp": {"input": 0.00, "output": 0.00},  # Free tier
    "gemini-1.5-pro": {"input": 1.25, "output": 5.00},
    "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
    # Default for unknown models
    "default": {"input": 1.00, "output": 3.00},
}


@dataclass
class TokenUsage:
    """Tracks token usage for a single LLM call."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


@dataclass
class CostEstimate:
    """Cost estimate for token usage."""

    input_cost: float = 0.0
    output_cost: float = 0.0
    total_cost: float = 0.0
    currency: str = "USD"

    def __add__(self, other: CostEstimate) -> CostEstimate:
        """Add two cost estimates."""
        return CostEstimate(
            input_cost=self.input_cost + other.input_cost,
            output_cost=self.output_cost + other.output_cost,
            total_cost=self.total_cost + other.total_cost,
            currency=self.currency,
        )


@dataclass
class TraceMetadata:
    """Metadata for a trace span."""

    trace_id: UUID = field(default_factory=uuid4)
    span_id: UUID = field(default_factory=uuid4)
    parent_span_id: UUID | None = None
    name: str = ""
    start_time: datetime = field(default_factory=lambda: datetime.now(UTC))
    end_time: datetime | None = None
    model: str | None = None
    provider: str | None = None
    token_usage: TokenUsage = field(default_factory=TokenUsage)
    cost: CostEstimate = field(default_factory=CostEstimate)
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def calculate_cost(model: str, usage: TokenUsage) -> CostEstimate:
    """Calculate the cost for token usage.

    Args:
        model: The model name.
        usage: The token usage.

    Returns:
        Cost estimate in USD.
    """
    pricing = TOKEN_PRICING.get(model, TOKEN_PRICING["default"])

    input_cost = (usage.input_tokens / 1_000_000) * pricing["input"]
    output_cost = (usage.output_tokens / 1_000_000) * pricing["output"]

    return CostEstimate(
        input_cost=input_cost,
        output_cost=output_cost,
        total_cost=input_cost + output_cost,
    )


class TelemetryManager:
    """Manages telemetry and observability.

    Provides:
    - LangSmith integration when available
    - Local SQLite fallback for offline tracing
    - Cost tracking and estimation
    - Performance metrics
    """

    def __init__(self) -> None:
        """Initialize the telemetry manager."""
        self._enabled = False
        self._langsmith_configured = False
        self._current_trace: TraceMetadata | None = None
        self._trace_stack: list[TraceMetadata] = []
        self._session_usage = TokenUsage()
        self._session_cost = CostEstimate()
        self._traces: list[TraceMetadata] = []

    def configure(
        self,
        *,
        enabled: bool = True,
        langsmith_api_key: str | None = None,
        project_name: str = "aphelion",
    ) -> None:
        """Configure telemetry settings.

        Args:
            enabled: Whether telemetry is enabled.
            langsmith_api_key: LangSmith API key.
            project_name: LangSmith project name.
        """
        self._enabled = enabled

        if enabled and langsmith_api_key:
            # Configure LangSmith environment variables
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_API_KEY"] = langsmith_api_key
            os.environ["LANGCHAIN_PROJECT"] = project_name
            self._langsmith_configured = True
            logger.info("LangSmith telemetry configured", project=project_name)
        else:
            os.environ.pop("LANGCHAIN_TRACING_V2", None)
            self._langsmith_configured = False
            if enabled:
                logger.info("Local-only telemetry enabled (no LangSmith)")

    @property
    def is_enabled(self) -> bool:
        """Check if telemetry is enabled."""
        return self._enabled

    @property
    def is_langsmith_configured(self) -> bool:
        """Check if LangSmith is configured."""
        return self._langsmith_configured

    @contextmanager
    def trace(
        self,
        name: str,
        *,
        model: str | None = None,
        provider: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Generator[TraceMetadata, None, None]:
        """Context manager for tracing a span.

        Args:
            name: Name of the span.
            model: Model being used.
            provider: Provider being used.
            metadata: Additional metadata.

        Yields:
            The trace metadata for this span.
        """
        if not self._enabled:
            yield TraceMetadata(name=name)
            return

        parent = self._trace_stack[-1] if self._trace_stack else None

        trace = TraceMetadata(
            name=name,
            parent_span_id=parent.span_id if parent else None,
            trace_id=parent.trace_id if parent else uuid4(),
            model=model,
            provider=provider,
            metadata=metadata or {},
        )

        self._trace_stack.append(trace)

        try:
            yield trace
        except Exception as e:
            trace.error = str(e)
            raise
        finally:
            trace.end_time = datetime.now(UTC)

            # Calculate cost if we have token usage
            if trace.model and trace.token_usage.total_tokens > 0:
                trace.cost = calculate_cost(trace.model, trace.token_usage)
                self._session_usage.input_tokens += trace.token_usage.input_tokens
                self._session_usage.output_tokens += trace.token_usage.output_tokens
                self._session_usage.total_tokens += trace.token_usage.total_tokens
                self._session_cost = self._session_cost + trace.cost

            self._traces.append(trace)
            self._trace_stack.pop()

            # Log the trace
            duration = (trace.end_time - trace.start_time).total_seconds()
            logger.info(
                "Trace completed",
                name=trace.name,
                duration_ms=round(duration * 1000, 2),
                tokens=trace.token_usage.total_tokens,
                cost_usd=round(trace.cost.total_cost, 6),
                error=trace.error,
            )

    @asynccontextmanager
    async def trace_async(
        self,
        name: str,
        *,
        model: str | None = None,
        provider: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AsyncGenerator[TraceMetadata, None]:
        """Async context manager for tracing a span.

        Args:
            name: Name of the span.
            model: Model being used.
            provider: Provider being used.
            metadata: Additional metadata.

        Yields:
            The trace metadata for this span.
        """
        with self.trace(name, model=model, provider=provider, metadata=metadata) as t:
            yield t

    def record_usage(
        self,
        trace: TraceMetadata,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> None:
        """Record token usage for a trace.

        Args:
            trace: The trace to record usage for.
            input_tokens: Number of input tokens.
            output_tokens: Number of output tokens.
        """
        trace.token_usage.input_tokens = input_tokens
        trace.token_usage.output_tokens = output_tokens
        trace.token_usage.total_tokens = input_tokens + output_tokens

    def get_session_usage(self) -> TokenUsage:
        """Get total token usage for this session."""
        return self._session_usage

    def get_session_cost(self) -> CostEstimate:
        """Get total cost for this session."""
        return self._session_cost

    def get_recent_traces(self, limit: int = 100) -> list[TraceMetadata]:
        """Get recent traces.

        Args:
            limit: Maximum number of traces to return.

        Returns:
            List of recent traces.
        """
        return self._traces[-limit:]

    def reset_session(self) -> None:
        """Reset session statistics."""
        self._session_usage = TokenUsage()
        self._session_cost = CostEstimate()
        self._traces.clear()

    def get_langchain_callbacks(self) -> list[BaseCallbackHandler]:
        """Get LangChain callback handlers for tracing.

        Returns:
            List of callback handlers.
        """
        callbacks: list[BaseCallbackHandler] = []

        if self._langsmith_configured:
            try:
                from langchain_core.tracers import LangChainTracer
                from langsmith import Client

                client = Client()
                tracer = LangChainTracer(client=client)
                callbacks.append(tracer)
            except ImportError:
                logger.warning("LangSmith not installed, skipping tracer")

        return callbacks


# Global telemetry manager instance
telemetry = TelemetryManager()


async def init_telemetry(
    *,
    enabled: bool = True,
    langsmith_api_key: str | None = None,
    project_name: str = "aphelion",
) -> None:
    """Initialize telemetry.

    This should be called during application startup.

    Args:
        enabled: Whether to enable telemetry.
        langsmith_api_key: Optional LangSmith API key.
        project_name: LangSmith project name.
    """
    telemetry.configure(
        enabled=enabled,
        langsmith_api_key=langsmith_api_key,
        project_name=project_name,
    )

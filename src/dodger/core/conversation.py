"""Conversation manager for chat sessions.

This module provides:
- Conversation lifecycle management
- Message history with context window handling
- Automatic title generation
- Integration with database persistence
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

import structlog

from dodger.config import config_manager
from dodger.core.telemetry import telemetry
from dodger.database import Conversation
from dodger.database.repository import (
    get_conversation_repository,
    get_message_repository,
)
from dodger.providers.base import ChatMessage, ChatResponse, StreamChunk
from dodger.providers.manager import provider_manager

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

logger = structlog.get_logger(__name__)

# Constants
FIRST_EXCHANGE_MESSAGE_COUNT = 2
MAX_TITLE_LENGTH = 50
MAX_PREVIEW_CONTENT_LENGTH = 200


@dataclass
class ConversationState:
    """In-memory state for an active conversation.

    Attributes:
        id: Unique conversation identifier.
        title: Conversation title.
        messages: List of messages in memory.
        system_prompt: Optional system prompt override.
        model: The model being used.
        provider: The provider being used.
        created_at: When the conversation started.
        is_dirty: Whether there are unsaved changes.
    """

    id: UUID = field(default_factory=uuid4)
    title: str = "New Conversation"
    messages: list[ChatMessage] = field(default_factory=list)
    system_prompt: str | None = None
    model: str | None = None
    provider: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    is_dirty: bool = False
    total_tokens: int = 0
    total_cost: float = 0.0


# Default system prompt
DEFAULT_SYSTEM_PROMPT = """You are a helpful, harmless, and honest AI assistant. \
You provide clear, accurate, and thoughtful responses. \
When you don't know something, you say so. \
You can help with a wide range of tasks including analysis, writing, coding, and creative work."""


class ConversationManager:
    """Manages chat conversations with LLMs.

    Provides:
    - Conversation creation and loading
    - Message sending with streaming
    - Context window management
    - Auto-save functionality
    """

    def __init__(self) -> None:
        """Initialize the conversation manager."""
        self._active_conversation: ConversationState | None = None
        self._conversations: dict[UUID, ConversationState] = {}

    @property
    def active_conversation(self) -> ConversationState | None:
        """Get the currently active conversation."""
        return self._active_conversation

    def create_conversation(
        self,
        title: str = "New Conversation",
        system_prompt: str | None = None,
        model: str | None = None,
    ) -> ConversationState:
        """Create a new conversation.

        Args:
            title: The conversation title.
            system_prompt: Optional system prompt.
            model: The model to use.

        Returns:
            The new conversation state.
        """
        settings = config_manager.settings

        conversation = ConversationState(
            title=title,
            system_prompt=system_prompt or DEFAULT_SYSTEM_PROMPT,
            model=model or settings.active_model,
            provider=settings.active_provider.value,
        )

        self._conversations[conversation.id] = conversation
        self._active_conversation = conversation

        logger.info(
            "Created conversation",
            id=str(conversation.id),
            title=title,
        )

        return conversation

    def set_active_conversation(self, conversation_id: UUID) -> ConversationState | None:
        """Set the active conversation by ID.

        Args:
            conversation_id: The conversation ID.

        Returns:
            The conversation state, or None if not found.
        """
        if conversation_id in self._conversations:
            self._active_conversation = self._conversations[conversation_id]
            return self._active_conversation
        return None

    async def load_conversation(self, conversation_id: UUID) -> ConversationState | None:
        """Load a conversation from the database.

        Args:
            conversation_id: The conversation ID.

        Returns:
            The loaded conversation state, or None if not found.
        """
        repo = None
        try:
            repo = await get_conversation_repository()
            db_conversation = await repo.get_with_messages(conversation_id)

            if not db_conversation:
                return None

            # Convert to in-memory state
            messages = [
                ChatMessage(role=msg.role, content=msg.content)
                for msg in db_conversation.messages
            ]

            conversation = ConversationState(
                id=db_conversation.id,
                title=db_conversation.title,
                messages=messages,
                system_prompt=db_conversation.system_prompt,
                model=db_conversation.model,
                provider=db_conversation.provider,
                created_at=db_conversation.created_at,
                total_tokens=db_conversation.total_tokens,
                total_cost=db_conversation.total_cost,
            )

            self._conversations[conversation.id] = conversation
            self._active_conversation = conversation

            logger.info(
                "Loaded conversation",
                id=str(conversation_id),
                message_count=len(messages),
            )

            return conversation

        except Exception:
            logger.exception("Failed to load conversation")
            return None
        finally:
            if repo:
                await repo.close()

    async def save_conversation(
        self, conversation: ConversationState | None = None
    ) -> bool:
        """Save a conversation to the database.

        Args:
            conversation: The conversation to save (defaults to active).

        Returns:
            True if saved successfully.
        """
        conversation = conversation or self._active_conversation
        if not conversation:
            return False

        conv_repo = None
        msg_repo = None
        try:
            conv_repo = await get_conversation_repository()
            msg_repo = await get_message_repository()

            # Check if conversation exists
            existing = await conv_repo.get_by_id(conversation.id)

            if existing:
                # Update existing
                existing.title = conversation.title
                existing.system_prompt = conversation.system_prompt
                existing.model = conversation.model
                existing.provider = conversation.provider
                existing.total_tokens = conversation.total_tokens
                existing.total_cost = conversation.total_cost
                # Update cached values
                existing.cached_message_count = len(conversation.messages)
                if conversation.messages:
                    last_content = conversation.messages[-1].content
                    existing.last_message_preview = last_content[:200] if len(last_content) > 200 else last_content
                await conv_repo.update(existing)
            else:
                # Create new
                last_preview = None
                if conversation.messages:
                    last_content = conversation.messages[-1].content
                    last_preview = last_content[:200] if len(last_content) > 200 else last_content

                db_conversation = Conversation(
                    id=conversation.id,
                    title=conversation.title,
                    system_prompt=conversation.system_prompt,
                    model=conversation.model,
                    provider=conversation.provider,
                    total_tokens=conversation.total_tokens,
                    total_cost=conversation.total_cost,
                    cached_message_count=len(conversation.messages),
                    last_message_preview=last_preview,
                )
                await conv_repo.create(db_conversation)

                # Save messages
                for msg in conversation.messages:
                    await msg_repo.add_to_conversation(
                        conversation_id=conversation.id,
                        role=msg.role,
                        content=msg.content,
                    )

            conversation.is_dirty = False
            logger.info("Saved conversation", id=str(conversation.id))
            return True

        except Exception:
            logger.exception("Failed to save conversation")
            return False
        finally:
            if conv_repo:
                await conv_repo.close()
            if msg_repo:
                await msg_repo.close()

    async def send_message(
        self,
        content: str,
        images: list[str] | None = None,
        conversation: ConversationState | None = None,
    ) -> ChatResponse:
        """Send a message and get a response.

        Args:
            content: The message content.
            images: Optional list of image URLs or base64 data.
            conversation: The conversation to use (defaults to active).

        Returns:
            The assistant's response.
        """
        conversation = conversation or self._active_conversation
        if not conversation:
            conversation = self.create_conversation()

        # Add user message
        user_message = ChatMessage(
            role="user",
            content=content,
            images=images or [],
        )
        conversation.messages.append(user_message)
        conversation.is_dirty = True

        # Build messages for API
        api_messages = self._build_api_messages(conversation)

        # Get response
        response = await provider_manager.chat(
            messages=api_messages,
            model=conversation.model,
        )

        # Add assistant message
        assistant_message = ChatMessage(
            role="assistant",
            content=response.content,
        )
        conversation.messages.append(assistant_message)

        # Update stats
        conversation.total_tokens += response.total_tokens
        conversation.total_cost += telemetry.get_session_cost().total_cost

        # Auto-save if enabled
        settings = config_manager.settings
        if settings.auto_save_conversations:
            await self.save_conversation(conversation)

        # Generate title if first exchange
        if (
            len(conversation.messages) == FIRST_EXCHANGE_MESSAGE_COUNT
            and conversation.title == "New Conversation"
        ):
            await self._generate_title(conversation)

        return response

    async def stream_message(
        self,
        content: str,
        images: list[str] | None = None,
        conversation: ConversationState | None = None,
    ) -> AsyncGenerator[StreamChunk, None]:
        """Stream a message response.

        Args:
            content: The message content.
            images: Optional list of image URLs or base64 data.
            conversation: The conversation to use (defaults to active).

        Yields:
            Response chunks as they're generated.
        """
        conversation = conversation or self._active_conversation
        if not conversation:
            conversation = self.create_conversation()

        # Ensure we have a model set
        settings = config_manager.settings
        model_to_use = conversation.model or settings.active_model
        
        logger.info(
            "Streaming message",
            model=model_to_use,
            provider=settings.active_provider.value,
            content_length=len(content),
        )

        # Add user message
        user_message = ChatMessage(
            role="user",
            content=content,
            images=images or [],
        )
        conversation.messages.append(user_message)
        conversation.is_dirty = True

        # Build messages for API
        api_messages = self._build_api_messages(conversation)
        
        logger.info("Calling stream_chat", message_count=len(api_messages), model=model_to_use)

        # Stream response
        full_response = ""
        total_input = 0
        total_output = 0

        async for chunk in provider_manager.stream_chat(
            messages=api_messages,
            model=model_to_use,  # Use the resolved model
        ):
            if chunk.content:
                full_response += chunk.content
            if chunk.is_final:
                total_input = chunk.input_tokens
                total_output = chunk.output_tokens
            yield chunk

        # Add assistant message
        assistant_message = ChatMessage(
            role="assistant",
            content=full_response,
        )
        conversation.messages.append(assistant_message)

        # Update stats
        conversation.total_tokens += total_input + total_output

        # Auto-save if enabled
        if settings.auto_save_conversations:
            await self.save_conversation(conversation)

        # Generate title if first exchange
        if (
            len(conversation.messages) == FIRST_EXCHANGE_MESSAGE_COUNT
            and conversation.title == "New Conversation"
        ):
            await self._generate_title(conversation)

    def _build_api_messages(self, conversation: ConversationState) -> list[ChatMessage]:
        """Build the message list for API calls.

        Includes system prompt and handles context window limits.

        Args:
            conversation: The conversation state.

        Returns:
            List of messages for the API.
        """
        messages: list[ChatMessage] = []

        # Add system prompt
        if conversation.system_prompt:
            messages.append(
                ChatMessage(role="system", content=conversation.system_prompt)
            )

        # Get max context messages from settings
        settings = config_manager.settings
        max_messages = settings.max_context_messages

        # Add conversation messages (respecting limit)
        conversation_messages = conversation.messages[-max_messages:]
        messages.extend(conversation_messages)

        return messages

    async def _generate_title(self, conversation: ConversationState) -> None:
        """Generate a title for the conversation based on content.

        Args:
            conversation: The conversation to title.
        """
        try:
            # Use a quick prompt to generate title
            title_prompt = [
                ChatMessage(
                    role="system",
                    content="Generate a short, descriptive title (max 6 words) "
                    "for this conversation. Reply with only the title, "
                    "no quotes or punctuation.",
                ),
                ChatMessage(
                    role="user",
                    content=(
                        f"User: {conversation.messages[0].content}\n\n"
                        f"Assistant: {conversation.messages[1].content[:MAX_PREVIEW_CONTENT_LENGTH]}"
                    ),
                ),
            ]

            response = await provider_manager.chat(
                messages=title_prompt,
                temperature=0.3,
                max_tokens=20,
            )

            title = response.content.strip().strip('"\'')
            if title:
                conversation.title = title[:MAX_TITLE_LENGTH]
                conversation.is_dirty = True
                await self.save_conversation(conversation)
                logger.info("Generated title", title=title)

        except Exception as e:
            logger.warning("Failed to generate title", error=str(e))

    async def get_recent_conversations(
        self, limit: int = 50
    ) -> list[dict[str, Any]]:
        """Get recent conversations for the sidebar.

        Args:
            limit: Maximum number of conversations.

        Returns:
            List of conversation summaries.
        """
        repo = None
        try:
            repo = await get_conversation_repository()
            conversations = await repo.get_recent(limit=limit)

            return [
                {
                    "id": str(conv.id),
                    "title": conv.title,
                    "preview": conv.preview,
                    "updated_at": conv.updated_at.isoformat(),
                    "is_pinned": conv.is_pinned,
                    "message_count": conv.message_count,
                }
                for conv in conversations
            ]

        except Exception:
            logger.exception("Failed to get recent conversations")
            return []
        finally:
            if repo:
                await repo.close()

    async def search_conversations(self, query: str) -> list[dict[str, Any]]:
        """Search conversations.

        Args:
            query: The search query.

        Returns:
            List of matching conversation summaries.
        """
        repo = None
        try:
            repo = await get_conversation_repository()
            conversations = await repo.search(query)

            return [
                {
                    "id": str(conv.id),
                    "title": conv.title,
                    "preview": conv.preview,
                    "updated_at": conv.updated_at.isoformat(),
                }
                for conv in conversations
            ]

        except Exception:
            logger.exception("Failed to search conversations")
            return []
        finally:
            if repo:
                await repo.close()

    async def delete_conversation(self, conversation_id: UUID) -> bool:
        """Delete a conversation.

        Args:
            conversation_id: The conversation to delete.

        Returns:
            True if deleted successfully.
        """
        repo = None
        try:
            repo = await get_conversation_repository()
            conversation = await repo.get_by_id(conversation_id)

            if conversation:
                await repo.delete(conversation)

                # Remove from memory
                if conversation_id in self._conversations:
                    del self._conversations[conversation_id]

                if (
                    self._active_conversation
                    and self._active_conversation.id == conversation_id
                ):
                    self._active_conversation = None

                logger.info("Deleted conversation", id=str(conversation_id))
                return True

            return False

        except Exception:
            logger.exception("Failed to delete conversation")
            return False
        finally:
            if repo:
                await repo.close()

    def clear_active_conversation(self) -> None:
        """Clear the active conversation (start fresh)."""
        self._active_conversation = None


# Global conversation manager instance
conversation_manager = ConversationManager()

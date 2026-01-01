"""Configuration management for Aphelion Magnetar.

This module provides:
- Pydantic-based settings management
- Secure API key storage with encryption
- Auto-detection of local LLM services
- Persistent user preferences
"""

from __future__ import annotations

import base64
import json
import os
import secrets
from enum import Enum
from pathlib import Path
from typing import Any

import httpx
import structlog
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from pydantic import BaseModel, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = structlog.get_logger(__name__)


class ProviderType(str, Enum):
    """Supported LLM provider types."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    OPENROUTER = "openrouter"
    OLLAMA = "ollama"
    LMSTUDIO = "lmstudio"


class ThemeMode(str, Enum):
    """Application theme modes."""

    LIGHT = "light"
    DARK = "dark"
    SYSTEM = "system"


class ProviderConfig(BaseModel):
    """Configuration for a single LLM provider."""

    enabled: bool = False
    api_key: SecretStr | None = None
    base_url: str | None = None
    default_model: str | None = None
    available_models: list[str] = Field(default_factory=list)

    model_config = {"validate_assignment": True}


class TelemetryConfig(BaseModel):
    """LangSmith telemetry configuration."""

    enabled: bool = False
    api_key: SecretStr | None = None
    project_name: str = "dodger"
    trace_all: bool = True


class ServerConfig(BaseModel):
    """Remote access server configuration."""

    enabled: bool = False
    host: str = "0.0.0.0"  # noqa: S104
    port: int = 8000
    api_token: SecretStr | None = None
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])


class AppearanceConfig(BaseModel):
    """UI appearance settings."""

    theme: ThemeMode = ThemeMode.DARK
    font_size: int = Field(default=14, ge=10, le=24)
    message_density: str = "comfortable"  # compact, comfortable, spacious
    accent_color: str = "#6366f1"  # Indigo-500
    show_avatars: bool = True
    show_timestamps: bool = True

    @field_validator("accent_color")
    @classmethod
    def validate_hex_color(cls, v: str) -> str:
        """Validate hex color format."""
        if not v.startswith("#") or len(v) != 7:
            msg = "Accent color must be a valid hex color (e.g., #6366f1)"
            raise ValueError(msg)
        return v


class Settings(BaseSettings):
    """Main application settings.

    Settings are loaded from:
    1. Environment variables (highest priority)
    2. Config file (~/.config/dodger/config.json)
    3. Default values (lowest priority)
    """

    model_config = SettingsConfigDict(
        env_prefix="APHELION_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    # Provider configurations
    providers: dict[ProviderType, ProviderConfig] = Field(
        default_factory=lambda: {
            ProviderType.OPENAI: ProviderConfig(
                base_url="https://api.openai.com/v1",
                default_model="gpt-4o",
            ),
            ProviderType.ANTHROPIC: ProviderConfig(
                base_url="https://api.anthropic.com",
                default_model="claude-3-5-sonnet-20241022",
            ),
            ProviderType.GOOGLE: ProviderConfig(
                default_model="gemini-2.0-flash-exp",
            ),
            ProviderType.OPENROUTER: ProviderConfig(
                base_url="https://openrouter.ai/api/v1",
                default_model="anthropic/claude-3.5-sonnet",
            ),
            ProviderType.OLLAMA: ProviderConfig(
                base_url="http://localhost:11434",
                enabled=False,  # Auto-detected
            ),
            ProviderType.LMSTUDIO: ProviderConfig(
                base_url="http://localhost:1234/v1",
                enabled=False,  # Auto-detected
            ),
        }
    )

    # Telemetry
    telemetry: TelemetryConfig = Field(default_factory=TelemetryConfig)

    # Server
    server: ServerConfig = Field(default_factory=ServerConfig)

    # Appearance
    appearance: AppearanceConfig = Field(default_factory=AppearanceConfig)

    # General settings
    active_provider: ProviderType = ProviderType.OPENAI
    active_model: str | None = None
    max_context_messages: int = Field(default=50, ge=1, le=500)
    stream_responses: bool = True
    auto_save_conversations: bool = True

    # Database
    database_path: Path = Field(
        default_factory=lambda: get_config_dir() / "conversations.db"
    )


def get_config_dir() -> Path:
    """Get the configuration directory, creating it if needed.

    Returns:
        Path to the configuration directory.
    """
    # Check if running on Android
    if "ANDROID_ROOT" in os.environ or "ANDROID_DATA" in os.environ:
        try:
            from dodger.core.android import get_android_config_dir
            config_dir = get_android_config_dir()
            config_dir.mkdir(parents=True, exist_ok=True)
            return config_dir
        except ImportError:
            pass

    if os.name == "nt":  # Windows
        config_dir = Path(os.environ.get("APPDATA", "~")) / "Aphelion"
    else:  # macOS/Linux
        config_dir = Path(os.environ.get("XDG_CONFIG_HOME", "~/.config")) / "dodger"

    config_dir = config_dir.expanduser()
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir


def get_data_dir() -> Path:
    """Get the data directory for larger files.

    Returns:
        Path to the data directory.
    """
    # Check if running on Android
    if "ANDROID_ROOT" in os.environ or "ANDROID_DATA" in os.environ:
        try:
            from dodger.core.android import get_android_data_dir
            data_dir = get_android_data_dir()
            data_dir.mkdir(parents=True, exist_ok=True)
            return data_dir
        except ImportError:
            pass

    if os.name == "nt":  # Windows
        data_dir = Path(os.environ.get("LOCALAPPDATA", "~")) / "Aphelion"
    else:  # macOS/Linux
        data_dir = Path(os.environ.get("XDG_DATA_HOME", "~/.local/share")) / "dodger"

    data_dir = data_dir.expanduser()
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


class SecureKeyStorage:
    """Secure storage for API keys using Fernet encryption.

    Keys are encrypted using a machine-specific key derived from
    a random salt stored alongside the encrypted data.
    """

    def __init__(self, storage_path: Path | None = None) -> None:
        """Initialize secure storage.

        Args:
            storage_path: Path to store encrypted keys. Defaults to config dir.
        """
        self.storage_path = storage_path or get_config_dir() / "keys.enc"
        self._fernet: Fernet | None = None
        self._salt: bytes | None = None

    def _get_machine_id(self) -> bytes:
        """Get a machine-specific identifier for key derivation."""
        # Use a combination of factors for machine identity
        import platform
        import uuid

        machine_info = f"{platform.node()}-{uuid.getnode()}"
        return machine_info.encode()

    def _derive_key(self, salt: bytes) -> bytes:
        """Derive an encryption key from machine ID and salt."""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=480000,
        )
        return base64.urlsafe_b64encode(kdf.derive(self._get_machine_id()))

    def _initialize_fernet(self) -> Fernet:
        """Initialize or load the Fernet cipher."""
        if self._fernet is not None:
            return self._fernet

        salt_path = self.storage_path.with_suffix(".salt")

        if salt_path.exists():
            self._salt = salt_path.read_bytes()
        else:
            self._salt = secrets.token_bytes(16)
            salt_path.write_bytes(self._salt)
            salt_path.chmod(0o600)

        key = self._derive_key(self._salt)
        self._fernet = Fernet(key)
        return self._fernet

    def store_key(self, provider: ProviderType, api_key: str) -> None:
        """Store an encrypted API key.

        Args:
            provider: The provider type.
            api_key: The API key to store.
        """
        fernet = self._initialize_fernet()

        # Load existing keys
        keys = self._load_keys()
        keys[provider.value] = api_key

        # Encrypt and save
        encrypted = fernet.encrypt(json.dumps(keys).encode())
        self.storage_path.write_bytes(encrypted)
        self.storage_path.chmod(0o600)

        logger.info("Stored API key", provider=provider.value)

    def get_key(self, provider: ProviderType) -> str | None:
        """Retrieve a decrypted API key.

        Args:
            provider: The provider type.

        Returns:
            The decrypted API key, or None if not found.
        """
        keys = self._load_keys()
        return keys.get(provider.value)

    def delete_key(self, provider: ProviderType) -> None:
        """Delete a stored API key.

        Args:
            provider: The provider type.
        """
        fernet = self._initialize_fernet()
        keys = self._load_keys()

        if provider.value in keys:
            del keys[provider.value]
            encrypted = fernet.encrypt(json.dumps(keys).encode())
            self.storage_path.write_bytes(encrypted)
            logger.info("Deleted API key", provider=provider.value)

    def _load_keys(self) -> dict[str, str]:
        """Load and decrypt all stored keys."""
        if not self.storage_path.exists():
            return {}

        fernet = self._initialize_fernet()
        try:
            encrypted = self.storage_path.read_bytes()
            decrypted = fernet.decrypt(encrypted)
            return json.loads(decrypted.decode())  # type: ignore[no-any-return]
        except Exception:
            logger.warning("Failed to decrypt keys, returning empty")
            return {}


class LocalLLMDetector:
    """Auto-detect locally running LLM services."""

    OLLAMA_HEALTH_ENDPOINT = "/api/tags"
    LMSTUDIO_HEALTH_ENDPOINT = "/v1/models"

    @classmethod
    async def detect_ollama(cls, base_url: str = "http://localhost:11434") -> bool:
        """Check if Ollama is running.

        Args:
            base_url: The base URL to check.

        Returns:
            True if Ollama is running and accessible.
        """
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                response = await client.get(f"{base_url}{cls.OLLAMA_HEALTH_ENDPOINT}")
                return response.status_code == 200
        except Exception:
            return False

    @classmethod
    async def detect_lmstudio(cls, base_url: str = "http://localhost:1234") -> bool:
        """Check if LMStudio is running.

        Args:
            base_url: The base URL to check.

        Returns:
            True if LMStudio is running and accessible.
        """
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                response = await client.get(f"{base_url}{cls.LMSTUDIO_HEALTH_ENDPOINT}")
                return response.status_code == 200
        except Exception:
            return False

    @classmethod
    async def get_ollama_models(cls, base_url: str = "http://localhost:11434") -> list[str]:
        """Get available Ollama models.

        Args:
            base_url: The Ollama base URL.

        Returns:
            List of available model names.
        """
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{base_url}{cls.OLLAMA_HEALTH_ENDPOINT}")
                if response.status_code == 200:
                    data = response.json()
                    return [model["name"] for model in data.get("models", [])]
        except Exception:
            pass
        return []

    @classmethod
    async def get_lmstudio_models(cls, base_url: str = "http://localhost:1234") -> list[str]:
        """Get available LMStudio models.

        Args:
            base_url: The LMStudio base URL.

        Returns:
            List of available model names.
        """
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{base_url}{cls.LMSTUDIO_HEALTH_ENDPOINT}")
                if response.status_code == 200:
                    data = response.json()
                    return [model["id"] for model in data.get("data", [])]
        except Exception:
            pass
        return []


class ConfigManager:
    """Manages application configuration with persistence."""

    def __init__(self) -> None:
        """Initialize the configuration manager."""
        self.config_path = get_config_dir() / "config.json"
        self.key_storage = SecureKeyStorage()
        self._settings: Settings | None = None

    def load(self) -> Settings:
        """Load settings from disk and environment.

        Returns:
            The loaded settings.
        """
        config_data: dict[str, Any] = {}

        # Load from config file if exists
        if self.config_path.exists():
            try:
                config_data = json.loads(self.config_path.read_text())
                logger.info("Loaded config from file", path=str(self.config_path))
            except Exception as e:
                logger.warning("Failed to load config file", error=str(e))

        # Create settings (will also read from environment)
        self._settings = Settings(**config_data)

        # Load API keys from secure storage
        self._load_api_keys()

        return self._settings

    def save(self) -> None:
        """Save current settings to disk."""
        if self._settings is None:
            return

        # Convert to dict, excluding secrets (they get masked as **** in json mode)
        config_data = self._settings.model_dump(
            mode="json",
            exclude={
                "providers": {"__all__": {"api_key"}},
                "server": {"api_token"},
                "telemetry": {"api_key"},
            },
        )

        # Manually add back secret values that should be persisted
        if self._settings.server.api_token:
            config_data["server"]["api_token"] = self._settings.server.api_token.get_secret_value()
        if self._settings.telemetry.api_key:
            config_data["telemetry"]["api_key"] = self._settings.telemetry.api_key.get_secret_value()

        # Save config file
        self.config_path.write_text(json.dumps(config_data, indent=2, default=str))
        self.config_path.chmod(0o600)
        logger.info("Saved config to file", path=str(self.config_path))

        # Save API keys to secure storage
        self._save_api_keys()

    def _load_api_keys(self) -> None:
        """Load API keys from secure storage into settings."""
        if self._settings is None:
            return

        for provider_type in ProviderType:
            key = self.key_storage.get_key(provider_type)
            if key:
                self._settings.providers[provider_type].api_key = SecretStr(key)
                self._settings.providers[provider_type].enabled = True

    def _save_api_keys(self) -> None:
        """Save API keys from settings to secure storage."""
        if self._settings is None:
            return

        for provider_type, config in self._settings.providers.items():
            if config.api_key:
                self.key_storage.store_key(provider_type, config.api_key.get_secret_value())

    @property
    def settings(self) -> Settings:
        """Get current settings, loading if necessary."""
        if self._settings is None:
            return self.load()
        return self._settings

    def update_provider(
        self,
        provider: ProviderType,
        *,
        enabled: bool | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        default_model: str | None = None,
        available_models: list[str] | None = None,
    ) -> None:
        """Update a provider's configuration.

        Args:
            provider: The provider to update.
            enabled: Whether the provider is enabled.
            api_key: The API key.
            base_url: The base URL (pass empty string to clear).
            default_model: The default model.
            available_models: List of available models.
        """
        config = self.settings.providers[provider]

        if enabled is not None:
            config.enabled = enabled
        if api_key is not None:
            config.api_key = SecretStr(api_key)
        # Allow setting base_url even if empty (to use default)
        if base_url is not None:
            config.base_url = base_url if base_url else None
        if default_model is not None:
            config.default_model = default_model
        if available_models is not None:
            config.available_models = available_models

        logger.info(
            "Updated provider config",
            provider=provider.value,
            base_url=config.base_url,
            default_model=config.default_model,
        )
        self.save()

    async def auto_detect_local_llms(self) -> dict[ProviderType, bool]:
        """Auto-detect locally running LLM services.

        Returns:
            Dictionary mapping provider types to detection results.
        """
        results: dict[ProviderType, bool] = {}

        # Check Ollama
        ollama_config = self.settings.providers[ProviderType.OLLAMA]
        ollama_url = ollama_config.base_url or "http://localhost:11434"
        results[ProviderType.OLLAMA] = await LocalLLMDetector.detect_ollama(ollama_url)

        if results[ProviderType.OLLAMA]:
            models = await LocalLLMDetector.get_ollama_models(ollama_url)
            ollama_config.available_models = models
            ollama_config.enabled = True
            if models and not ollama_config.default_model:
                ollama_config.default_model = models[0]
            logger.info("Detected Ollama", models=models)

        # Check LMStudio
        lmstudio_config = self.settings.providers[ProviderType.LMSTUDIO]
        lmstudio_url = lmstudio_config.base_url or "http://localhost:1234"
        results[ProviderType.LMSTUDIO] = await LocalLLMDetector.detect_lmstudio(lmstudio_url)

        if results[ProviderType.LMSTUDIO]:
            models = await LocalLLMDetector.get_lmstudio_models(lmstudio_url)
            lmstudio_config.available_models = models
            lmstudio_config.enabled = True
            if models and not lmstudio_config.default_model:
                lmstudio_config.default_model = models[0]
            logger.info("Detected LMStudio", models=models)

        self.save()
        return results


# Global configuration manager instance
config_manager = ConfigManager()

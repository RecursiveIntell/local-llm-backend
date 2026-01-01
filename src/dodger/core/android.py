"""Android platform utilities for Dodger.

This module provides Android-specific functionality and platform detection
for cross-platform compatibility when deploying via pyside6-android-deploy.
"""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)


@lru_cache(maxsize=1)
def is_android() -> bool:
    """Check if running on Android.

    Returns:
        True if running on Android platform.
    """
    # Android sets ANDROID_ROOT and ANDROID_DATA environment variables
    return "ANDROID_ROOT" in os.environ or "ANDROID_DATA" in os.environ


def get_android_data_dir() -> Path:
    """Get the appropriate data directory for Android.

    On Android, apps should store data in app-specific directories.

    Returns:
        Path to the data directory.
    """
    if is_android():
        # Android app internal storage
        android_data = os.environ.get("ANDROID_DATA", "/data")
        # Typically: /data/data/com.dodger.app/files
        app_files = Path(android_data) / "data" / "com.dodger.app" / "files"
        if app_files.exists():
            return app_files
        # Fallback to home directory if available
        home = Path.home()
        if home.exists():
            data_dir = home / ".dodger"
            data_dir.mkdir(parents=True, exist_ok=True)
            return data_dir
    # Non-Android: use standard path
    return Path.home() / ".local" / "share" / "dodger"


def get_android_config_dir() -> Path:
    """Get the appropriate config directory for Android.

    Returns:
        Path to the config directory.
    """
    if is_android():
        return get_android_data_dir() / "config"
    return Path.home() / ".config" / "dodger"


def get_android_cache_dir() -> Path:
    """Get the appropriate cache directory for Android.

    Returns:
        Path to the cache directory.
    """
    if is_android():
        android_data = os.environ.get("ANDROID_DATA", "/data")
        cache_dir = Path(android_data) / "data" / "com.dodger.app" / "cache"
        if cache_dir.exists():
            return cache_dir
        return get_android_data_dir() / "cache"
    return Path.home() / ".cache" / "dodger"


def get_display_metrics() -> dict[str, float]:
    """Get display metrics for UI scaling on Android.

    Returns:
        Dictionary with display metrics (dpi, scale_factor).
    """
    if is_android():
        # On Android, Qt handles DPI scaling automatically
        # These are reasonable defaults for mobile
        return {
            "dpi": 160.0,  # mdpi baseline
            "scale_factor": 1.0,
            "is_mobile": True,
        }
    return {
        "dpi": 96.0,  # Desktop baseline
        "scale_factor": 1.0,
        "is_mobile": False,
    }


def setup_android_environment() -> None:
    """Configure environment for Android if needed.

    Call this early in application startup.
    """
    if not is_android():
        return

    logger.info("Detected Android platform, configuring environment")

    # Ensure directories exist
    for dir_func in [get_android_data_dir, get_android_config_dir, get_android_cache_dir]:
        try:
            dir_path = dir_func()
            dir_path.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.warning("Failed to create directory", path=str(dir_path), error=str(e))

    # Set environment variables for other libraries
    os.environ.setdefault("HOME", str(get_android_data_dir()))

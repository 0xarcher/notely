"""Centralized prompt registry for managing LLM prompts.

This module provides a PyTorch-style registry pattern for managing
prompts across the application. Prompts are registered once and
can be retrieved by name.
"""

from __future__ import annotations

import logging
import threading
from typing import ClassVar

logger = logging.getLogger(__name__)


class PromptRegistry:
    """Centralized registry for LLM prompts.

    This class uses the registry pattern (similar to PyTorch's module registry)
    to manage prompts in a centralized location. Prompts are stored as templates
    that can be formatted with variables at runtime.

    Thread-safe: All operations are protected by a lock for concurrent access.

    Example:
        # Register a prompt
        PromptRegistry.register("greeting", "Hello {name}!")

        # Retrieve and format
        prompt = PromptRegistry.get("greeting").format(name="Alice")
    """

    _prompts: ClassVar[dict[str, str]] = {}
    _lock: ClassVar[threading.Lock] = threading.Lock()

    @classmethod
    def register(cls, name: str, prompt: str, allow_overwrite: bool = False) -> None:
        """Register a prompt template.

        Args:
            name: Unique identifier for the prompt
            prompt: Prompt template string (can contain {variables})
            allow_overwrite: If True, allow overwriting existing prompts.
                           If False (default), raise ValueError on duplicate.

        Raises:
            ValueError: If prompt already exists and allow_overwrite=False

        Example:
            PromptRegistry.register("summarize", "Summarize: {text}")
            PromptRegistry.register("summarize", "New prompt", allow_overwrite=True)
        """
        with cls._lock:
            if name in cls._prompts and not allow_overwrite:
                raise ValueError(
                    f"Prompt '{name}' already exists. Use allow_overwrite=True to replace it."
                )

            if name in cls._prompts:
                logger.warning(f"Overwriting existing prompt: {name}")

            cls._prompts[name] = prompt
            logger.debug(f"Registered prompt: {name}")

    @classmethod
    def get(cls, name: str) -> str:
        """Retrieve a prompt template by name.

        Args:
            name: Prompt identifier

        Returns:
            Prompt template string

        Raises:
            KeyError: If prompt not found

        Example:
            template = PromptRegistry.get("summarize")
            prompt = template.format(text="Hello world")
        """
        with cls._lock:
            if name not in cls._prompts:
                raise KeyError(
                    f"Prompt '{name}' not found. "
                    f"Available prompts: {', '.join(cls._prompts.keys())}"
                )

            return cls._prompts[name]

    @classmethod
    def list(cls) -> list[str]:
        """List all registered prompt names.

        Returns:
            List of prompt names

        Example:
            prompts = PromptRegistry.list()
            print(f"Available prompts: {prompts}")
        """
        with cls._lock:
            return list(cls._prompts.keys())

    @classmethod
    def clear(cls) -> None:
        """Clear all registered prompts.

        This is primarily useful for testing.
        """
        with cls._lock:
            cls._prompts.clear()
            logger.debug("Cleared all prompts from registry")
